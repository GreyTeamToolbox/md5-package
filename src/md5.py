#!/usr/bin/env python

"""
md5.py.

A custom implementation of the MD5 hashing algorithm in Python, designed to handle both string and file inputs.
This script provides a modular and detailed approach to MD5 hashing, including padding, chunk processing,
and state manipulation, all in line with the MD5 specification.

Modules:
    - left_rotate: Function for left rotating a 32-bit integer.
    - process_md5_chunk: Function to process each 64-byte chunk of the message.
    - apply_md5_padding: Function to apply MD5 padding to the input data.
    - read_file_in_chunks: Generator function to read a file in 64-byte chunks.
    - process_data: Function to handle data input, whether from a file or a string, and apply padding.
    - md5: Main function to compute the MD5 hash of a given input string or file.
    - format_md5: Function to format the final MD5 hash in hexadecimal format.
"""

import os
import struct
import argparse

from typing import Generator, Union


# Initial MD5 hash values
INITIAL_STATE_A = 0x67452301
INITIAL_STATE_B = 0xefcdab89
INITIAL_STATE_C = 0x98badcfe
INITIAL_STATE_D = 0x10325476

# MD5 Rotation amounts
ROTATION_AMOUNTS = [
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
    5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20,
    4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
    6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21
]

# Precomputed sine function constants
SINE_CONSTANTS = [int(4294967296 * abs(__import__('math').sin(i + 1))) & 0xFFFFFFFF for i in range(64)]


class MD5Exception(Exception):
    """Custom exception for MD5 hashing errors."""


def left_rotate(value: int, shift_amount: int) -> int:
    """
    Perform a left circular rotation on a 32-bit integer by a specified number of bits.

    This is a helper function used in the MD5 algorithm to shift the bits of an integer
    to the left, with the bits wrapping around to the right side.

    Arguments:
        value (int): The integer to be rotated.
        shift_amount (int): The number of positions to rotate the bits.

    Returns:
        int: The result of the left rotation operation, masked to 32 bits.
    """
    return ((value << shift_amount) | (value >> (32 - shift_amount))) & 0xFFFFFFFF


def process_md5_chunk(chunk: bytes, state_a: int, state_b: int, state_c: int, state_d: int) -> tuple[int, int, int, int]:
    """
    Process a 64-byte chunk of data for the MD5 algorithm, updating the hash state.

    This function takes a 64-byte chunk of the message (512 bits) and applies a series of
    transformations based on the MD5 algorithm. It uses four auxiliary functions (F, G, H, I)
    and predefined constants to modify the input state variables.

    Arguments:
        chunk (bytes): A 64-byte chunk of the message to be processed.
        state_a (int): The current state value A.
        state_b (int): The current state value B.
        state_c (int): The current state value C.
        state_d (int): The current state value D.

    Returns:
        tuple[int, int, int, int]: Updated state values (A, B, C, D) after processing the chunk.
    """
    message_schedule = struct.unpack('<16I', chunk)
    a, b, c, d = state_a, state_b, state_c, state_d

    for i in range(64):
        if 0 <= i <= 15:
            f = (b & c) | (~b & d)
            g = i
        elif 16 <= i <= 31:
            f = (d & b) | (~d & c)
            g = (5 * i + 1) % 16
        elif 32 <= i <= 47:
            f = b ^ c ^ d
            g = (3 * i + 5) % 16
        else:
            f = c ^ (b | ~d)
            g = (7 * i) % 16

        temp = (a + f + message_schedule[g] + SINE_CONSTANTS[i]) & 0xFFFFFFFF
        a, d, c, b = d, (b + left_rotate(temp, ROTATION_AMOUNTS[i])) & 0xFFFFFFFF, b, c

    return (state_a + a) & 0xFFFFFFFF, (state_b + b) & 0xFFFFFFFF, (state_c + c) & 0xFFFFFFFF, (state_d + d) & 0xFFFFFFFF


def apply_md5_padding(data: bytes, original_length: int) -> bytes:
    """
    Apply MD5-specific padding to data to ensure it is a multiple of 512 bits (64 bytes).

    This function pads the data with a '1' bit followed by zero bits, until the length is 56 bytes
    less than a multiple of 64. Then, the original message length is appended as a 64-bit integer.

    Arguments:
        data (bytes): The input data to be padded.
        original_length (int): The length of the original data in bytes.

    Returns:
        bytes: The padded data as per MD5 specifications.
    """
    padded_data = bytearray(data)
    padded_data.append(0x80)  # Append the bit '1' as per MD5 padding requirements
    while len(padded_data) % 64 != 56:
        padded_data.append(0)  # Append '0' bits

    padded_data += struct.pack('<Q', original_length * 8)
    return bytes(padded_data)


def read_file_in_chunks(file_path: str) -> Generator[bytes, None, None]:
    """
    Read a file in 64-byte chunks, yielding each chunk for processing.

    Reads through the file in chunks of 64 bytes (512 bits) and yields each chunk
    for processing, along with the total length of the file.

    Arguments:
        file_path (str): The path to the file to be read.

    Yields:
        tuple[bytes, int]: Each chunk of data and the total file length.
    """
    total_length = 0
    last_chunk = b''
    with open(file_path, 'rb') as file:
        while True:
            chunk = file.read(64)
            chunk_length = len(chunk)
            total_length += chunk_length

            if chunk_length < 64:
                last_chunk = chunk
                break
            yield chunk
    yield last_chunk, total_length


def process_data(input_data: Union[str, bytes], is_file: bool = False) -> bytes:
    """
    Process input data and return padded data for MD5 hashing.

    Handles both file and string inputs by reading the data, applying the MD5-specific
    padding, and returning the padded data ready for processing.

    Arguments:
        input_data (Union[str, bytes]): The data to be processed, which could be a string or file path.
        is_file (bool): A flag indicating whether the input data is a file.

    Returns:
        bytes: The padded data ready for MD5 hashing.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        MD5Exception: If an invalid input type is provided.
    """
    if is_file:
        if not os.path.isfile(input_data):
            raise FileNotFoundError(f"File not found: {input_data}")

        padded_data = bytearray()
        total_length = 0

        for chunk in read_file_in_chunks(input_data):
            if isinstance(chunk, tuple):
                last_chunk, total_length = chunk
                padded_data += apply_md5_padding(last_chunk, total_length)
            else:
                padded_data += chunk

        return bytes(padded_data)

    if isinstance(input_data, str):
        input_data = bytearray(input_data, 'utf-8')
    elif not isinstance(input_data, (bytes, bytearray)):
        raise MD5Exception("Invalid input type; expected str, bytes, or bytearray.")

    total_length = len(input_data)
    return apply_md5_padding(input_data, total_length)


def md5(input_data: Union[str, bytes], is_file: bool = False) -> str:
    """
    Generate an MD5 hash for a given string or file.

    This function calculates the MD5 hash of the provided data by processing each 64-byte chunk
    of the input, updating the MD5 state, and then formatting the final hash in hexadecimal.

    Arguments:
        input_data (Union[str, bytes]): The input string, file path, or byte data.
        is_file (bool): Flag to indicate whether the input_data is a file path.

    Returns:
        str: The MD5 hash in hexadecimal format.

    Raises:
        MD5Exception: If an error occurs during hashing.
        FileNotFoundError: If the specified file does not exist.
    """
    # Initial state values for MD5
    state_a, state_b, state_c, state_d = INITIAL_STATE_A, INITIAL_STATE_B, INITIAL_STATE_C, INITIAL_STATE_D

    # Process and pad the data
    padded_data = process_data(input_data, is_file=is_file)

    # Process each 64-byte chunk in the padded data
    for i in range(0, len(padded_data), 64):
        state_a, state_b, state_c, state_d = process_md5_chunk(padded_data[i:i + 64], state_a, state_b, state_c, state_d)

    return format_md5(state_a, state_b, state_c, state_d)


def format_md5(state_a: int, state_b: int, state_c: int, state_d: int) -> str:
    """
    Format the final MD5 hash from the state values.

    After the MD5 state values have been computed, they are packed into a byte string
    and formatted as a hexadecimal hash string.

    Arguments:
        state_a (int): The final state value A.
        state_b (int): The final state value B.
        state_c (int): The final state value C.
        state_d (int): The final state value D.

    Returns:
        str: The MD5 hash in hexadecimal format.
    """
    digest = struct.pack('<4I', state_a, state_b, state_c, state_d)
    return ''.join(f'{byte:02x}' for byte in digest)  # noqa


def main():
    """Parse arguments and execute the MD5 hashing."""
    parser = argparse.ArgumentParser(description="MD5 hash generator for strings and files.")
    parser.add_argument("input", type=str, help="The input string or file path to hash.")
    parser.add_argument("-f", "--file", action="store_true", help="Specify if the input is a file.")

    args = parser.parse_args()

    try:
        result = md5(args.input, is_file=args.file)
        print(f"MD5 Hash: {result}")
    except MD5Exception as e:
        print(f"Error: {e}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
