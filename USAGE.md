# Using saltminer

## Build

```
cargo build --release
```

Needs `ocl-icd-opencl-dev` and a GPU ICD (`intel-opencl-icd`, NVIDIA driver, etc).

## Run

```
saltminer \
  --deployer      0xFactoryAddress \
  --initcode-hash 0xKeccakOfInitCode \
  --args-hash     0xKeccakOfEncodedArgs \
  --mask          0xffffffff00000000000000000000000000000000 \
  --match         0x0000000000000000000000000000000000000000
```

On a hit it prints `salt` and `home` and exits. Hand that `salt` to your factory.

## Two GPUs

```
saltminer --shard 0/2 --device 0:0  ...
saltminer --shard 1/2 --device 0:1  ...
```

`saltminer --list-devices` to find the indices.

## Everything else

`saltminer --help` for the full flag list. [README.md](README.md) for how it works and the EIP-1167 hash gotcha.
