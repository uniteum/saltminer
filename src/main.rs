use anyhow::{bail, Context, Result};
use clap::Parser;
use ocl::{flags, Buffer, Context as OclContext, Device, Kernel, Platform, Program, Queue};
use saltminer::{
    address_from_state, compute_base_state, compute_lane_masks, parse_hex_bytes, parse_shard,
    parse_u64,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

const KERNEL_SRC: &str = include_str!("kernel.cl");

#[derive(Parser, Debug)]
#[command(
    name = "saltminer",
    about = "GPU-accelerated CREATE2 salt miner. See README.md for design."
)]
struct Args {
    /// List available OpenCL platforms and devices and exit.
    #[arg(long = "listdevices")]
    list_devices: bool,

    /// OpenCL device, as platform_index:device_index (see --listdevices).
    #[arg(long, default_value = "0:0")]
    device: String,

    /// Factory (deployer) address, 20-byte hex.
    #[arg(long)]
    deployer: Option<String>,

    /// keccak256 of the init code the factory will deploy, 32-byte hex.
    #[arg(long = "initcodehash")]
    initcode_hash: Option<String>,

    /// keccak256(abi.encode(...)) of the parameters bound into the factory's salt, 32-byte hex.
    #[arg(long = "argshash")]
    args_hash: Option<String>,

    /// Address bit-mask, 20-byte hex (160 bits).
    #[arg(long)]
    mask: Option<String>,

    /// Address target value, 20-byte hex (160 bits). Only bits where mask == 1 are compared.
    #[arg(long = "target")]
    target: Option<String>,

    /// Minimum salt (inclusive). u64 decimal or 0x-prefixed hex.
    #[arg(long, default_value = "0")]
    min: String,

    /// Maximum salt (exclusive). u64 decimal or 0x-prefixed hex.
    #[arg(long, default_value = "0xffffffffffffffff")]
    max: String,

    /// Shard as worker_index/worker_count (e.g. 0/1, 2/4).
    #[arg(long, default_value = "0/1")]
    shard: String,

    /// Threads per kernel dispatch.
    #[arg(long = "dispatch", default_value_t = 1 << 20)]
    global_size: usize,
}

fn list_devices() -> Result<()> {
    for (pi, p) in Platform::list().into_iter().enumerate() {
        let pname = p.name().unwrap_or_else(|_| "<unknown>".into());
        println!("Platform {pi}: {pname}");
        match Device::list_all(&p) {
            Ok(devices) => {
                for (di, d) in devices.into_iter().enumerate() {
                    let dname = d.name().unwrap_or_else(|_| "<unknown>".into());
                    println!("  Device {pi}:{di} -- {dname}");
                }
            }
            Err(e) => println!("  (failed to enumerate devices: {e})"),
        }
    }
    Ok(())
}

fn select_device(spec: &str) -> Result<(Platform, Device)> {
    let (pi, di) = spec.split_once(':').context("device must be platform:device")?;
    let pi: usize = pi.parse().context("platform index not a number")?;
    let di: usize = di.parse().context("device index not a number")?;
    let platforms = Platform::list();
    let platform = *platforms
        .get(pi)
        .with_context(|| format!("no OpenCL platform at index {pi}"))?;
    let devices = Device::list_all(&platform)?;
    let device = *devices
        .get(di)
        .with_context(|| format!("no device at index {di} on platform {pi}"))?;
    Ok((platform, device))
}

fn main() -> Result<()> {
    let args = Args::parse();

    if args.list_devices {
        return list_devices();
    }

    let deployer =
        parse_hex_bytes::<20>(args.deployer.as_deref().context("--deployer required")?)?;
    let initcode_hash = parse_hex_bytes::<32>(
        args.initcode_hash
            .as_deref()
            .context("--initcodehash required")?,
    )?;
    let args_hash =
        parse_hex_bytes::<32>(args.args_hash.as_deref().context("--argshash required")?)?;
    let mask = parse_hex_bytes::<20>(args.mask.as_deref().context("--mask required")?)?;
    let mut target =
        parse_hex_bytes::<20>(args.target.as_deref().context("--target required")?)?;
    // AND with the mask so callers can pass high-entropy values (e.g. the digits
    // of pi) and have non-mask bits silently dropped instead of making the
    // comparison unsatisfiable.
    for i in 0..20 {
        target[i] &= mask[i];
    }
    let min = parse_u64(&args.min)?;
    let max = parse_u64(&args.max)?;
    let (w, n) = parse_shard(&args.shard)?;
    if min >= max {
        bail!("--min must be < --max");
    }
    let global_size = args.global_size;
    if global_size == 0 {
        bail!("--dispatch must be > 0");
    }

    let (platform, device) = select_device(&args.device)?;
    let dname = device.name().unwrap_or_else(|_| "<unknown>".into());
    eprintln!("saltminer: using device {} -- {}", args.device, dname);
    eprintln!(
        "saltminer: shard {}/{}, range [{:#x}, {:#x}), global_size {}",
        w, n, min, max, global_size
    );

    let base_state = compute_base_state(&deployer, &args_hash, &initcode_hash);
    let mask_lanes = compute_lane_masks(&mask, &target);

    let ctx = OclContext::builder()
        .platform(platform)
        .devices(device)
        .build()?;
    let queue = Queue::new(&ctx, device, None)?;
    let program = Program::builder()
        .src(KERNEL_SRC)
        .devices(device)
        .build(&ctx)?;

    let base_state_buf = Buffer::<u64>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_READ_ONLY | flags::MEM_COPY_HOST_PTR)
        .len(17)
        .copy_host_slice(&base_state)
        .build()?;
    let mask_buf = Buffer::<u64>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_READ_ONLY | flags::MEM_COPY_HOST_PTR)
        .len(6)
        .copy_host_slice(&mask_lanes)
        .build()?;
    let found_buf = Buffer::<i32>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_READ_WRITE | flags::MEM_COPY_HOST_PTR)
        .len(1)
        .copy_host_slice(&[0i32])
        .build()?;
    let result_buf = Buffer::<u64>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_READ_WRITE | flags::MEM_COPY_HOST_PTR)
        .len(4)
        .copy_host_slice(&[0u64; 4])
        .build()?;

    let stop = Arc::new(AtomicBool::new(false));
    {
        let stop = stop.clone();
        ctrlc::set_handler(move || stop.store(true, Ordering::SeqCst))
            .context("installing Ctrl-C handler")?;
    }

    // First salt this worker tests is min + w. Each dispatch advances by
    // global_size * n to stay on this worker's residue class mod n.
    let mut start_salt = min
        .checked_add(w)
        .context("min + shard offset overflows u64")?;
    let step = (global_size as u64)
        .checked_mul(n)
        .context("global_size * shard count overflows u64")?;

    let started = Instant::now();
    let mut tested: u128 = 0;

    while start_salt < max && !stop.load(Ordering::SeqCst) {
        let kernel = Kernel::builder()
            .program(&program)
            .name("mine")
            .queue(queue.clone())
            .global_work_size(global_size)
            .arg(start_salt)
            .arg(n)
            .arg(max)
            .arg(&base_state_buf)
            .arg(&mask_buf)
            .arg(&found_buf)
            .arg(&result_buf)
            .build()?;
        unsafe {
            kernel.enq()?;
        }
        queue.finish()?;

        let mut found = [0i32];
        found_buf.read(&mut found[..]).enq()?;
        if found[0] != 0 {
            let mut res = [0u64; 4];
            result_buf.read(&mut res[..]).enq()?;
            let salt = res[0];
            let addr = address_from_state(res[1], res[2], res[3]);
            eprintln!();
            println!("salt = 0x{:064x}", salt);
            println!("home = 0x{}", hex::encode(addr));
            return Ok(());
        }

        tested += global_size as u128;
        let elapsed = started.elapsed().as_secs_f64().max(1e-9);
        eprint!(
            "\rsaltminer: tested {} salts, {:.2} Mhash/s, next {:#x}  ",
            tested,
            (tested as f64) / elapsed / 1e6,
            start_salt.saturating_add(step),
        );

        start_salt = match start_salt.checked_add(step) {
            Some(v) => v,
            None => break,
        };
    }
    eprintln!();

    let resume_min = start_salt.saturating_sub(w);
    if stop.load(Ordering::SeqCst) {
        eprintln!(
            "saltminer: interrupted. resume with --min {:#x} --max {:#x} --shard {}/{}",
            resume_min, max, w, n
        );
    } else {
        eprintln!(
            "saltminer: range exhausted without a match. widen with --min {:#x} --max <new> --shard {}/{}",
            resume_min, w, n
        );
    }
    std::process::exit(1);
}
