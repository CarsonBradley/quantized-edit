"""Verify the per-section sha256s in a .strata header without filling the page cache.

Same check as `strata-inkling --selfcheck --native PACK --verify-hash`, but on the CPU and with F_NOCACHE reads, so
hashing a 14 GB pack does not fill memory with cached file pages (which a memory watchdog may count as used).
  python3 -I eval/verify_strata_hash.py PACK.strata
"""
import fcntl, hashlib, os, struct, sys

F_NOCACHE = getattr(fcntl, "F_NOCACHE", 48)


def main(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        fcntl.fcntl(fd, F_NOCACHE, 1)
        hdr = os.pread(fd, 4096, 0)
        assert hdr[:8] == b"STRATA\x00\x01", "bad magic"
        flags, profile, nsec = struct.unpack_from("<IHH", hdr, 8)
        ok = True
        for i in range(nsec):
            off, length, sha, sid, sflags, _ = struct.unpack_from("<QQ32sIIQ", hdr, 0x10 + 64 * i)
            h = hashlib.sha256()
            pos, end = off, off + length
            while pos < end:
                b = os.pread(fd, min(64 << 20, end - pos), pos)
                if not b:
                    raise IOError("short read")
                h.update(b); pos += len(b)
            good = h.digest() == sha
            ok &= good
            print(f"section id={sid} off={off} len={length} flags={sflags}: {'OK' if good else 'MISMATCH'} {h.hexdigest()[:16]}", flush=True)
        print("verify-hash:", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    finally:
        os.close(fd)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
