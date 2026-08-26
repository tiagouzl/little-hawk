"""Tests for bench_ruler_eviction.extract_answer — harness correctness, not engine."""

import unicodedata

import pytest

from bench_ruler_eviction import extract_answer


def test_number_basic():
    assert extract_answer("resposta 123456 fim", "number") == "123456"


def test_number_no_false_positive_boundary():
    # \b não existe entre letra e dígito (ambos \w) → não deve casar
    assert extract_answer("ABC123456DEF", "number") is None


def test_number_last_occurrence():
    assert extract_answer("123456 e depois 654321 fim", "number") == "654321"


def test_word_nfc():
    assert extract_answer("resposta é botão fim", "word") == "botão"


def test_word_nfd():
    nfd = unicodedata.normalize("NFD", "botão")
    assert extract_answer(f"resposta é {nfd} fim", "word") == "botão"


def test_word_upper():
    assert extract_answer("resposta é Botão fim", "word") == "botão"


def test_word_nfd_upper():
    nfd = unicodedata.normalize("NFD", "Botão")
    assert extract_answer(f"resposta é {nfd} fim", "word") == "botão"


def test_date_nfd_and_case():
    nfd = unicodedata.normalize("NFD", "terça-feira")
    assert extract_answer(f"dia é {nfd} fim", "date") == "terça-feira"
    assert extract_answer("dia é Terça-feira fim", "date") == "terça-feira"
    assert extract_answer("dia é TERÇA-FEIRA fim", "date") == "terça-feira"
