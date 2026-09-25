import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseLabel
from pypdf import PdfReader
from docx import document
