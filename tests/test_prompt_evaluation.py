
import pytest

from app.main import extract_fields_from_text, validate_data


# Golden dataset: input text and the expected field values.
GOLDEN_CASES = [
    {
        "name": "employee details",
        "text": (
            "Employee Name: Harika Chamarthi\n"
            "Department: AI\n"
            "Email: harika@example.com"
        ),
        "fields": ["Employee Name", "Department", "Email"],
        "expected": {
            "Employee Name": "Harika Chamarthi",
            "Department": "AI",
            "Email": "harika@example.com",
        },
    },
    {
        "name": "missing department",
        "text": (
            "Employee Name: Ravi Kumar\n"
            "Email: ravi@example.com"
        ),
        "fields": ["Employee Name", "Department", "Email"],
        "expected": {
            "Employee Name": "Ravi Kumar",
            "Department": None,
            "Email": "ravi@example.com",
        },
    },
]


@pytest.mark.parametrize("case", GOLDEN_CASES, ids=lambda case: case["name"])
def test_extraction_matches_golden_dataset(case):
    actual = extract_fields_from_text(case["text"], case["fields"])

    for field, expected_value in case["expected"].items():
        assert actual.get(field) == expected_value, (
            f"Field {field!r}: expected {expected_value!r}, "
            f"got {actual.get(field)!r}"
        )


def test_missing_required_field_fails_validation():
    data = {
        "Employee Name": "Ravi Kumar",
        "Department": None,
        "Email": "ravi@example.com",
    }

    result = validate_data(
        data=data,
        required_fields=["Employee Name", "Department", "Email"],
    )

    assert result["is_valid"] is False
    assert result["missing_fields"] == ["Department"]
