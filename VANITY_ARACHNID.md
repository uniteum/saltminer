# Vanity Arachnid

Deploy a copy of Arachnid's deterministic deployment proxy at a vanity address, using `saltminer` to find the CREATE2 salt and the existing Arachnid proxy at `0x4e59b44847b379578588920cA78FbF26c0B4956C` as the launcher. The deployed contract is byte-identical to Arachnid's and lands at the same vanity address on every chain that has the proxy.

Arachnid's init code is:

```
0x604580600e600039806000f350fe7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffe03601600081602082378035828234f58015156039578182fd5b8082525050506014600cf3
```

and its `keccak256` is `0x50ea9137a35a9ad33b0ed4a431e9b6996ea9ed1f14781126cec78f168c0e64e5`. Arachnid doesn't mix parameters into its salt, so `--argshash` is zero and the full 32-byte CREATE2 salt is caller-controlled. See [README.md](README.md) for the mask/match model and the salt-space math.

## 1. Mine

Target 8 leading zero hex digits (4 zero bytes). ~4.3B attempts; seconds to a minute on a desktop GPU.

```
deployer=0x4e59b44847b379578588920cA78FbF26c0B4956C
initcodehash=0x50ea9137a35a9ad33b0ed4a431e9b6996ea9ed1f14781126cec78f168c0e64e5
argshash=0xE396da99091B535B65384914B178b9264c7426da000000000000000000000000
mask=0xffffff00000000000000000000000000000000ff
match=0x3141590000000000000000000000000000000097
saltminer --deployer $deployer --initcodehash $initcodehash --argshash $argshash --mask $mask --match $match
```

On a hit saltminer prints the `salt` and the predicted `home` address.
```
salt=0x00000000000000000000000000000000000000000000000000000000d711155f
```

## 2. Verify

```
cast create2 --deployer $deployer --salt $salt --init-code-hash $initcodehash
```

Must equal `home` exactly. If not, something is off — fix it before spending gas.

## 3. Deploy

Send a transaction to `0x4e59…` with calldata = `bytes32(salt) ‖ initcode`:

```
salt=0x000000000000000000000000000000000000000000000000000000000c7869ed
salt=0xE396da99091B535B65384914B178b9264c7426da00000000000000000c7869ed
deployer=0x4e59b44847b379578588920cA78FbF26c0B4956C
arachnid_code=0x604580600e600039806000f350fe7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffe03601600081602082378035828234f58015156039578182fd5b8082525050506014600cf3
vanity_data=$(cast concat-hex $salt $arachnid_code)
home=0x314159b4108323c48c6cc92f7ed84d4626de4f97
cast send $deployer $vanity_data --rpc-url 11155111 --private-key $tx_key
```

`cast code <home> --rpc-url <rpc>` should now return Arachnid's runtime bytecode.

## Other chains

Same `home` on every chain that has the Arachnid proxy — repeat step 3 with the same calldata. On chains that lack it, first replay Arachnid's presigned Nick's-method deployment transaction (fund `0x3fab184622dc19b6109349b94811493bf2a45362` with a bit of gas and broadcast the raw tx from the Arachnid repo), then step 3.
