# Vanity Arachnid

Deploy a copy of Arachnid's deterministic deployment proxy at a vanity address, using `saltminer` to find the CREATE2 salt and the existing Arachnid proxy at `0x4e59b44847b379578588920cA78FbF26c0B4956C` as the launcher. The deployed contract is byte-identical to Arachnid's and lands at the same vanity address on every chain that has the proxy.

Arachnid's init code is:

```
0x604580600e600039806000f350fe7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffe03601600081602082378035828234f58015156039578182fd5b8082525050506014600cf3
```

and its `keccak256` is `0x50ea9137a35a9ad33b0ed4a431e9b6996ea9ed1f14781126cec78f168c0e64e5`. Arachnid doesn't mix parameters into its salt, so `--args-hash` is zero and the full 32-byte CREATE2 salt is caller-controlled. See [README.md](README.md) for the mask/match model and the salt-space math.

## 1. Mine

Target 8 leading zero hex digits (4 zero bytes). ~4.3B attempts; seconds to a minute on a desktop GPU.

```
saltminer \
  --deployer      0x4e59b44847b379578588920cA78FbF26c0B4956C \
  --initcode-hash 0x50ea9137a35a9ad33b0ed4a431e9b6996ea9ed1f14781126cec78f168c0e64e5 \
  --args-hash     0x0000000000000000000000000000000000000000000000000000000000000000 \
  --mask          0xffffffff00000000000000000000000000000000 \
  --match         0x0000000000000000000000000000000000000000
```

On a hit saltminer prints the u64 `salt` and the predicted `home` address.

## 2. Verify

```
cast create2 \
  --deployer       0x4e59b44847b379578588920cA78FbF26c0B4956C \
  --salt           $(cast to-uint256 <saltU64>) \
  --init-code-hash 0x50ea9137a35a9ad33b0ed4a431e9b6996ea9ed1f14781126cec78f168c0e64e5
```

Must equal `home` exactly. If not, something is off — fix it before spending gas.

## 3. Deploy

Send a transaction to `0x4e59…` with calldata = `bytes32(salt) ‖ initcode`:

```
cast send 0x4e59b44847b379578588920cA78FbF26c0B4956C \
  $(cast concat-hex \
      $(cast to-uint256 <saltU64>) \
      0x604580600e600039806000f350fe7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffe03601600081602082378035828234f58015156039578182fd5b8082525050506014600cf3) \
  --rpc-url <rpc> --private-key <key>
```

`cast code <home> --rpc-url <rpc>` should now return Arachnid's runtime bytecode.

## Other chains

Same `home` on every chain that has the Arachnid proxy — repeat step 3 with the same calldata. On chains that lack it, first replay Arachnid's presigned Nick's-method deployment transaction (fund `0x3fab184622dc19b6109349b94811493bf2a45362` with a bit of gas and broadcast the raw tx from the Arachnid repo), then step 3.
