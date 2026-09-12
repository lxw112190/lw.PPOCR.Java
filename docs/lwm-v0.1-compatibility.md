# LWM v0.1 compatibility notes

This document records the binary contract implemented by the Java loader. The
authoritative writer and native validator are in `lw.PPOCR.C`; this file is a
compatibility record, not a new format proposal.

## Fixed layout

- Little-endian integers and IEEE-754 FP32 values.
- Header: 160 bytes.
- Tensor record: 80 bytes.
- Node record: 72 bytes.
- Section offsets are 8-byte aligned.
- The checksum is FNV-1a 64 over the entire file with header bytes 128..135
  treated as zero while computing it.
- v0.1 requires the no-memory-plan flag and does not accept a workspace plan.

Header offsets match `converter/lwm_v0.py` in `lw.PPOCR.C`:

| Offset | Field |
|---:|---|
| 0 | `LWM0` magic |
| 4 | format major/minor (`0.1`) |
| 8 | header size |
| 12 | flags |
| 16..28 | tensor/node/input/output counts |
| 32..112 | section offsets and sizes |
| 112 | exact file size |
| 120 | workspace size |
| 128 | content checksum |
| 136..159 | reserved, zero |

All input bytes are untrusted. The Java loader rejects malformed sections,
integer overflow, invalid dimensions, unsupported operators, unexpected
parameter sizes, non-constant payloads, and graph indexes outside the tensor
table before publishing `LwmModel`.

