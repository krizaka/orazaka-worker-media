#!/usr/bin/env python3
"""The Orazaka asset envelope, in Python. The other half of `EncryptedAssetService`.

Byte-for-byte the same format as the Java side, because the asset store is written by three
processes in two languages and a store only one of them can read is not a store. The layout lives
in ``docs/ASSET_ENCRYPTION.md``; this module and
``orazaka-libs/orazaka-ai-engine/orazaka-assets/.../EnvelopeCodec.java`` are its two implementations, and
``test_main.py`` asserts they agree on a fixture neither produced.

Envelope, not direct encryption: one data key per file, wrapped under a master key from the
keyring. Rotating the master key is a line in that file, not a rewrite of the store.

Block-addressable: AES-GCM's tag covers a whole message, so a file sealed as one message would
have to be decrypted in full to serve an HTTP ``Range``. The payload is a sequence of independently
sealed blocks, and a range decrypts the blocks it touches.
"""

import base64
import os
import struct

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

MAGIC = b"ORZAENC1"
VERSION = 1
ALG_AES_256_GCM = 1
DEFAULT_BLOCK_SIZE = 64 * 1024
NONCE_PREFIX_LENGTH = 8
TAG_LENGTH = 16
WRAP_NONCE_LENGTH = 12
KEY_BYTES = 32


class EnvelopeError(Exception):
    """The file is not an envelope, is truncated, or did not authenticate."""


class MasterKeyring:
    """The local keyring, read from the same file the Java side reads.

    Deliberately the same shape a KMS has — several keys, one active, addressed by id — so the
    local path is not a different code path from the one that matters.
    """

    def __init__(self, path: str):
        self.path = path
        self.keys = {}
        self.active = None
        if not os.path.isfile(path):
            raise EnvelopeError(
                f"master keyring not found at {path}; assets cannot be written or read "
                "without it. Create one with `orazaka assets keygen`."
            )
        mode = os.stat(path).st_mode & 0o077
        if mode:
            raise EnvelopeError(
                f"the master keyring at {path} is readable beyond its owner; run: chmod 600 {path}"
            )
        with open(path, encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                name, _, value = line.partition("=")
                name, value = name.strip(), value.strip()
                if name == "active":
                    self.active = value
                else:
                    material = base64.b64decode(value)
                    if len(material) != KEY_BYTES:
                        raise EnvelopeError(f"master key '{name}' is not 32 bytes")
                    self.keys[name] = material
        if not self.keys or self.active not in self.keys:
            raise EnvelopeError(f"the master keyring at {path} names no usable active key")

    def wrap(self, data_key: bytes) -> tuple:
        """Wraps a data key under the active master key. Returns ``(key_id, blob)``."""
        nonce = os.urandom(WRAP_NONCE_LENGTH)
        # The key id is authenticated with the key, so a header that renames it fails to unwrap.
        sealed = AESGCM(self.keys[self.active]).encrypt(
            nonce, data_key, self.active.encode("utf-8")
        )
        return self.active, nonce + sealed

    def unwrap(self, key_id: str, wrapped: bytes) -> bytes:
        """Unwraps a data key, or refuses. Never falls back to treating the file as plaintext."""
        master = self.keys.get(key_id)
        if master is None:
            raise EnvelopeError(
                f"this deployment holds no master key '{key_id}'; the asset cannot be opened here"
            )
        try:
            return AESGCM(master).decrypt(
                wrapped[:WRAP_NONCE_LENGTH], wrapped[WRAP_NONCE_LENGTH:], key_id.encode("utf-8")
            )
        except Exception as failure:
            raise EnvelopeError(
                f"the wrapped data key did not authenticate under master key '{key_id}'"
            ) from failure


def build_header(block_size: int, plain_length: int, key_id: str, wrapped: bytes,
                 nonce_prefix: bytes) -> bytes:
    """The header bytes, which are also the AAD prefix every block is bound to."""
    key_id_bytes = key_id.encode("utf-8")
    return b"".join([
        MAGIC,
        struct.pack(">BB", VERSION, ALG_AES_256_GCM),
        struct.pack(">I", block_size),
        struct.pack(">Q", plain_length),
        struct.pack(">H", len(key_id_bytes)), key_id_bytes,
        struct.pack(">H", len(wrapped)), wrapped,
        nonce_prefix,
    ])


def parse_header(blob: bytes) -> dict:
    """Parses a header from the front of ``blob``."""
    if len(blob) < 24 or blob[:8] != MAGIC:
        raise EnvelopeError("not an Orazaka envelope")
    version, algorithm = struct.unpack(">BB", blob[8:10])
    if version != VERSION:
        raise EnvelopeError(f"unsupported envelope version: {version}")
    if algorithm != ALG_AES_256_GCM:
        raise EnvelopeError(f"unsupported envelope algorithm: {algorithm}")
    block_size = struct.unpack(">I", blob[10:14])[0]
    plain_length = struct.unpack(">Q", blob[14:22])[0]
    key_id_length = struct.unpack(">H", blob[22:24])[0]
    at = 24
    key_id = blob[at:at + key_id_length].decode("utf-8"); at += key_id_length
    wrapped_length = struct.unpack(">H", blob[at:at + 2])[0]; at += 2
    wrapped = blob[at:at + wrapped_length]; at += wrapped_length
    nonce_prefix = blob[at:at + NONCE_PREFIX_LENGTH]; at += NONCE_PREFIX_LENGTH
    if len(nonce_prefix) < NONCE_PREFIX_LENGTH:
        raise EnvelopeError("truncated envelope header")
    return {
        "version": version, "blockSize": block_size, "plainLength": plain_length,
        "keyId": key_id, "wrapped": wrapped, "noncePrefix": nonce_prefix, "headerLength": at,
    }


def _nonce(nonce_prefix: bytes, index: int) -> bytes:
    return nonce_prefix + struct.pack(">I", index)


def _aad(header_bytes: bytes, index: int) -> bytes:
    return header_bytes + struct.pack(">I", index)


def is_encrypted(path: str) -> bool:
    """Whether this file is an envelope — the migrator's whole notion of 'already done'."""
    try:
        with open(path, "rb") as handle:
            return handle.read(len(MAGIC)) == MAGIC
    except OSError:
        return False


def encrypt_file(source: str, target: str, keyring: MasterKeyring,
                 block_size: int = DEFAULT_BLOCK_SIZE) -> None:
    """Seals ``source`` into ``target``, atomically.

    Written to a sibling temp and moved into place, so a process killed mid-write leaves the
    previous file — or none — and never a half-sealed one.
    """
    plain_length = os.path.getsize(source)
    data_key = os.urandom(KEY_BYTES)
    key_id, wrapped = keyring.wrap(data_key)
    nonce_prefix = os.urandom(NONCE_PREFIX_LENGTH)
    header = build_header(block_size, plain_length, key_id, wrapped, nonce_prefix)
    aesgcm = AESGCM(data_key)

    temp = target + ".orz-tmp"
    os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
    try:
        with open(source, "rb") as src, open(temp, "wb") as out:
            out.write(header)
            index = 0
            remaining = plain_length
            while remaining > 0:
                chunk = src.read(min(block_size, remaining))
                if len(chunk) == 0:
                    raise EnvelopeError("source ended early")
                out.write(aesgcm.encrypt(_nonce(nonce_prefix, index), chunk, _aad(header, index)))
                remaining -= len(chunk)
                index += 1
        os.replace(temp, target)
    finally:
        if os.path.exists(temp):
            os.remove(temp)


def encrypt_in_place(path: str, keyring: MasterKeyring,
                     block_size: int = DEFAULT_BLOCK_SIZE) -> bool:
    """Seals a plaintext file where it lies. Returns whether it did anything.

    Idempotent by construction, which is what makes a migration resumable: run it again.
    """
    if is_encrypted(path):
        return False
    encrypt_file(path, path, keyring, block_size)
    return True


def decrypt_bytes(path: str, keyring: MasterKeyring, offset: int = 0) -> bytes:
    """Opens an envelope from a plaintext offset, decrypting only the blocks it touches."""
    with open(path, "rb") as handle:
        blob = handle.read()
    header = parse_header(blob)
    header_bytes = build_header(header["blockSize"], header["plainLength"],
                                header["keyId"], header["wrapped"], header["noncePrefix"])
    data_key = keyring.unwrap(header["keyId"], header["wrapped"])
    aesgcm = AESGCM(data_key)

    block_size = header["blockSize"]
    sealed_size = block_size + TAG_LENGTH
    total = (header["plainLength"] + block_size - 1) // block_size
    first = min(offset // block_size, total) if block_size else 0

    out = bytearray()
    for index in range(first, total):
        start = header["headerLength"] + index * sealed_size
        plain_here = min(block_size, header["plainLength"] - index * block_size)
        sealed = blob[start:start + plain_here + TAG_LENGTH]
        try:
            out += aesgcm.decrypt(_nonce(header["noncePrefix"], index), sealed,
                                  _aad(header_bytes, index))
        except Exception as failure:
            raise EnvelopeError(f"block {index} did not authenticate") from failure
    return bytes(out[offset - first * block_size:])
