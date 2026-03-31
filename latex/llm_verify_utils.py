#!/usr/bin/env python3
"""Utilities for optional LLM-assisted LaTeX verification/refinement."""

import difflib
import json
import os

from dotenv import load_dotenv
from openai import OpenAI


def load_env(root_dir: str) -> None:
    """Load .env from the workspace root if present."""
    load_dotenv(os.path.join(root_dir, '.env'))


def llm_is_configured(root_dir: str) -> bool:
    load_env(root_dir)
    return bool(os.getenv('OPENAI_API_KEY'))


def generate_llm_verified_tex(
    *,
    root_dir: str,
    document_name: str,
    raw_tex: str,
    verified_tex: str,
    verification_items,
    evidence_summary: dict,
) -> str | None:
    """
    Ask an OpenAI model to produce a final LaTeX fragment.
    Returns None if API credentials are not configured.
    """
    load_env(root_dir)
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return None

    model = os.getenv('OPENAI_MODEL', 'gpt-4.1-mini')
    client = OpenAI(api_key=api_key)

    verification_payload = [
        {'status': status, 'claim': claim, 'detail': detail}
        for status, claim, detail in verification_items
    ]
    payload = {
        'document_name': document_name,
        'evidence_summary': evidence_summary,
        'verification_items': verification_payload,
        'raw_tex': raw_tex,
        'verified_tex': verified_tex,
    }

    instructions = (
        "You are refining a LaTeX analysis fragment for a robotics paper. "
        "Use the verified_tex as the authoritative starting point. "
        "Your job is to produce a final improved LaTeX fragment that preserves "
        "all factual numeric claims, keeps the same overall structure, and only "
        "strengthens wording when supported by the evidence_summary and verification_items. "
        "If a claim is Qualified, Contradicted, or External context, keep the wording cautious. "
        "Do not invent numbers, references, figures, citations, or experiments. "
        "Do not output markdown fences or commentary. Return only valid LaTeX fragment text."
    )

    try:
        response = client.responses.create(
            model=model,
            instructions=instructions,
            input=json.dumps(payload, ensure_ascii=False),
        )
    except Exception as exc:
        print(f'[WARN] LLM layer failed: {exc}')
        return None

    text = getattr(response, 'output_text', None)
    if text:
        return text.strip()

    parts = []
    for item in getattr(response, 'output', []):
        for content in getattr(item, 'content', []):
            if getattr(content, 'type', '') == 'output_text':
                parts.append(content.text)
    result = ''.join(parts).strip()
    return result or None


def generate_llm_text(
    *,
    root_dir: str,
    model: str | None = None,
    instructions: str,
    payload: dict,
) -> str | None:
    """
    Generic helper to obtain structured text from an OpenAI model.
    Returns None if credentials are missing or the request fails.
    """
    load_env(root_dir)
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return None

    model_name = model or os.getenv('OPENAI_MODEL', 'gpt-4.1-mini')
    client = OpenAI(api_key=api_key)

    try:
        response = client.responses.create(
            model=model_name,
            instructions=instructions,
            input=json.dumps(payload, ensure_ascii=False),
        )
    except Exception as exc:
        print(f'[WARN] LLM layer failed: {exc}')
        return None

    text = getattr(response, 'output_text', None)
    if text:
        return text.strip()

    parts = []
    for item in getattr(response, 'output', []):
        for content in getattr(item, 'content', []):
            if getattr(content, 'type', '') == 'output_text':
                parts.append(content.text)
    result = ''.join(parts).strip()
    return result or None


def generate_text_diff(original_text: str, revised_text: str, from_name: str, to_name: str) -> str:
    """Return a unified diff between two text blobs."""
    diff = difflib.unified_diff(
        original_text.splitlines(),
        revised_text.splitlines(),
        fromfile=from_name,
        tofile=to_name,
        lineterm='',
    )
    return '\n'.join(diff) + '\n'


def _latex_escape(text: str) -> str:
    return (text.replace('\\', r'\textbackslash{}')
                .replace('&', r'\&')
                .replace('%', r'\%')
                .replace('_', r'\_')
                .replace('#', r'\#')
                .replace('{', r'\{')
                .replace('}', r'\}')
                .replace('$', r'\$'))


def build_revision_trace_latex(
    original_text: str,
    revised_text: str,
    *,
    max_items: int = 12,
) -> str:
    """
    Build a readable LaTeX section summarizing how the LLM changed the verified text.
    """
    matcher = difflib.SequenceMatcher(
        None,
        original_text.splitlines(),
        revised_text.splitlines(),
    )
    items = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            continue
        before = '\n'.join(original_text.splitlines()[i1:i2]).strip()
        after = '\n'.join(revised_text.splitlines()[j1:j2]).strip()
        if not before and not after:
            continue
        items.append((tag, before, after))
        if len(items) >= max_items:
            break

    lines = [r'\subsubsection{LLM Revision Trace}']
    if not items:
        lines.append(
            'The LLM output matches the verified text at the tracked diff level, '
            'so no additional textual revisions were recorded.'
        )
        lines.append('')
        return '\n'.join(lines)

    lines.append(
        'The following trace summarizes the main textual edits introduced by the '
        'LLM layer relative to the verified version.'
    )
    lines.append('')
    for idx, (tag, before, after) in enumerate(items, start=1):
        lines.append(rf'\subsubsection{{Revision {idx}}}')
        lines.append(r'\textbf{Edit type:} ' + _latex_escape(tag.title()) + r'\\')
        if before:
            lines.append(r'\textbf{Before:}\\')
            lines.append(_latex_escape(before))
            lines.append('')
        if after:
            lines.append(r'\textbf{After:}\\')
            lines.append(_latex_escape(after))
            lines.append('')
    return '\n'.join(lines)
