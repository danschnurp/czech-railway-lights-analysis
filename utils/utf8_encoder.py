import base64
from typing import Union, Tuple


class UTF8StringEncoder:
    """
    A utility class for handling UTF-8 string encoding and decoding with various formats.
    """

    @staticmethod
    def encode_string(input_string: str,
                      output_format: str = 'bytes',
                      handle_errors: str = 'strict') -> Union[bytes, str]:
        """
        Encode a string to UTF-8.

        Args:
            input_string: The string to encode
            output_format: Format of the output ('bytes', 'hex', 'base64')
            handle_errors: How to handle encoding errors ('strict', 'ignore', 'replace')

        Returns:
            Encoded string in specified format

        Raises:
            ValueError: If invalid output_format is specified
            UnicodeEncodeError: If string cannot be encoded (in strict mode)
        """
        try:
            # First encode to UTF-8 bytes
            utf8_bytes = input_string.encode('utf-8', errors=handle_errors)

            # Convert to requested format
            if output_format == 'bytes':
                return utf8_bytes
            elif output_format == 'hex':
                return utf8_bytes.hex()
            elif output_format == 'base64':
                return base64.b64encode(utf8_bytes).decode('ascii')
            else:
                raise ValueError(f"Unsupported output format: {output_format}")

        except UnicodeEncodeError as e:
            if handle_errors == 'strict':
                raise
            elif handle_errors == 'ignore':
                return UTF8StringEncoder.encode_string(
                    ''.join(c for c in input_string if ord(c) < 0x10000),
                    output_format
                )
            else:  # replace
                return UTF8StringEncoder.encode_string(
                    input_string.encode('utf-8', 'replace').decode('utf-8'),
                    output_format
                )

    @staticmethod
    def decode_string(encoded_input: Union[bytes, str],
                      input_format: str = 'bytes',
                      handle_errors: str = 'strict') -> Tuple[str, bool]:
        """
        Decode a UTF-8 encoded string.

        Args:
            encoded_input: The encoded string to decode
            input_format: Format of the input ('bytes', 'hex', 'base64')
            handle_errors: How to handle decoding errors ('strict', 'ignore', 'replace')

        Returns:
            Tuple of (decoded_string, success_flag)

        Raises:
            ValueError: If invalid input_format is specified
        """
        try:
            # Convert from input format to bytes
            if input_format == 'bytes' and isinstance(encoded_input, bytes):
                utf8_bytes = encoded_input
            elif input_format == 'hex' and isinstance(encoded_input, str):
                utf8_bytes = bytes.fromhex(encoded_input)
            elif input_format == 'base64' and isinstance(encoded_input, str):
                utf8_bytes = base64.b64decode(encoded_input)
            else:
                raise ValueError(f"Invalid input format or type: {input_format}")

            # Decode bytes to string
            decoded_string = utf8_bytes.decode('utf-8', errors=handle_errors)
            return decoded_string, True

        except (UnicodeDecodeError, ValueError) as e:
            if handle_errors == 'strict':
                raise
            elif handle_errors == 'ignore':
                return encoded_input.decode('utf-8', 'ignore'), False
            else:  # replace
                return encoded_input.decode('utf-8', 'replace'), False

