"""Tests for the privacy.pii module."""

import pytest

from promptarchive.privacy.pii import PIIDetector, PIIFinding, PIIReport


class TestPIIDetector:
    def test_no_pii_clean_text(self):
        report = PIIDetector.scan("The contract was signed on Monday.")
        assert not report.has_pii
        assert report.findings == []

    def test_detects_email(self):
        report = PIIDetector.scan("Contact us at alice@example.com for details.")
        assert report.has_pii
        labels = [f.label for f in report.findings]
        assert "email" in labels

    def test_detects_phone(self):
        report = PIIDetector.scan("Call 555-867-5309 for support.")
        assert report.has_pii
        labels = [f.label for f in report.findings]
        assert "phone" in labels

    def test_detects_ssn(self):
        report = PIIDetector.scan("His SSN is 123-45-6789.")
        assert report.has_pii
        labels = [f.label for f in report.findings]
        assert "ssn" in labels

    def test_detects_ip_address(self):
        report = PIIDetector.scan("Server IP: 192.168.1.100")
        assert report.has_pii
        labels = [f.label for f in report.findings]
        assert "ip_address" in labels

    def test_detects_api_key(self):
        report = PIIDetector.scan("Authorization: Bearer sk-abcdef1234567890abcdef1234")
        assert report.has_pii
        labels = [f.label for f in report.findings]
        assert "api_key" in labels

    def test_multiple_pii_types(self):
        text = "Email alice@example.com or call 555-867-5309"
        report = PIIDetector.scan(text)
        assert report.has_pii
        assert len(report.findings) >= 2

    def test_redact_email(self):
        text = "Email me at bob@example.org please."
        redacted = PIIDetector.redact(text)
        assert "bob@example.org" not in redacted
        assert "[EMAIL]" in redacted

    def test_redact_phone(self):
        text = "Call 555-867-5309 now."
        redacted = PIIDetector.redact(text)
        assert "555-867-5309" not in redacted
        assert "[PHONE]" in redacted

    def test_redact_ip_address(self):
        text = "Connect to 10.0.0.1."
        redacted = PIIDetector.redact(text)
        assert "10.0.0.1" not in redacted
        assert "[IP_ADDRESS]" in redacted

    def test_redact_ssn(self):
        text = "SSN: 123-45-6789"
        redacted = PIIDetector.redact(text)
        assert "123-45-6789" not in redacted
        assert "[SSN]" in redacted

    def test_to_dict(self):
        report = PIIDetector.scan("alice@example.com")
        d = report.to_dict()
        assert d["has_pii"] is True
        assert d["finding_count"] >= 1
        assert isinstance(d["findings"], list)
        assert "label" in d["findings"][0]
        assert "value" in d["findings"][0]

    def test_finding_positions(self):
        text = "Contact alice@example.com today."
        report = PIIDetector.scan(text)
        assert report.has_pii
        for f in report.findings:
            assert text[f.start:f.end] == f.value

    def test_empty_text(self):
        report = PIIDetector.scan("")
        assert not report.has_pii

    def test_redact_leaves_non_pii_intact(self):
        text = "Hello, world. The weather is nice today."
        assert PIIDetector.redact(text) == text
