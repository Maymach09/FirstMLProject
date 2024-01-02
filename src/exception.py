import sys
import logging

def error_message_detail(error, exc_tb):
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error occurred in Python script: {file_name} line number: {exc_tb.tb_lineno} error message: {error}"
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message