# Background checkpoint offloading

Configure non-interactive SSH **on the training host**, including host key and
authentication. Then launch training with:

```bash
uv run moe_slm --offload-host my-laptop --offload-dir /home/me/moe-checkpoints
```

`my-laptop` can be an SSH config alias or `user@host`; its SSH configuration
controls the port, identity, and connection route. The destination directory
must be an absolute path on your machine. Check connectivity first with
`ssh -o BatchMode=yes my-laptop true` from the training host.

You can check the connection and destination before starting:

```bash
./prelaunch.sh --offload-host my-laptop --offload-dir /home/me/moe-checkpoints
```

The check is non-interactive and does not create files on the destination. If
the directory does not exist, it checks whether its nearest existing parent
appears writable. A failed SSH or destination check makes prelaunch return NO-GO.

For Tailscale, both machines need connectivity through the same tailnet (or a
working route), and your laptop needs an SSH server. Use `user@100.x.y.z` or
`user@machine.tailnet-name.ts.net`, not an `https://` URL. Find the laptop's
Tailscale IPv4 address on the laptop with `tailscale ip -4`. Make sure the
training host can authenticate non-interactively and the host key is known;
test SSH once interactively before using the prelaunch command.

The trainer writes checkpoints locally as before, then transfers `latest.pt`
and `best.pt` in the background. Existing local checkpoints are also queued at
startup. The destination is updated atomically only after a complete transfer.
If your machine is offline, the worker retries and keeps only the newest pending
snapshot of each name in `checkpoints/.offload/`. Keep that directory when
restarting the same training host: unsent snapshots are recovered when
offloading is enabled again. Normal shutdown waits approximately 10 seconds for
offloading; unsent snapshots remain queued for the next run. Local checkpoint
serialization and MLflow artifact logging still happen in the training process.
The destination's `latest.pt` and `best.pt` are overwritten by subsequent
successful transfers.
