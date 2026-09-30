import pytest
from src.agent.numeric_grounding import ungrounded_numbers, unbacked_narrative_claims

def test_numeric_grounding():
    evidence = [{"city": "Cairo", "revenue": 1234567.89, "growth": 0.156}]
    
    # Should pass
    assert ungrounded_numbers("The revenue in Cairo is 1.2M, with a growth of 15.6%. Top 5 branches.", evidence) == []
    
    # Should pass (Arabic digits)
    assert ungrounded_numbers("The revenue in Cairo is ١.٢M, with a growth of 15.6%.", evidence) == []
    
    # Should fail on 1.5M
    assert ungrounded_numbers("The revenue was 1.5M which is false.", evidence) == ['1.5M']
    
    # Should pass exact large number
    assert ungrounded_numbers("We saw 1,234,567 revenue.", evidence) == []
    
    # Should fail on 20%
    assert ungrounded_numbers("We saw 1,234,567 revenue and a 20% growth.", evidence) == ['20%']

def test_unbacked_narrative_claims():
    # Evidence with small drops and no share > 50%
    evidence_small = [
        {"restaurant_name": "Glens BakeHouse", "drop_amount": -4185},
        {"restaurant_name": "REKDEE", "drop_amount": -2672},
        {"restaurant_name": "The Belgian Waffle Cafe", "drop_amount": -2536}
    ]
    
    # False claim: claiming small drops represent the "bulk" of an 81.5M drop
    false_claim = "These three restaurants together represent the bulk of the overall decline."
    violations = unbacked_narrative_claims(false_claim, evidence_small)
    assert len(violations) > 0
    assert "Proportion hallucination" in violations[0]
    
    # Honest claim: does not use forbidden proportion words without proof
    honest_claim = "These are the top three decliners, with Glens BakeHouse dropping by 4,185."
    assert unbacked_narrative_claims(honest_claim, evidence_small) == []
    
    # Evidence WITH explicit > 50% share
    evidence_with_share = [
        {"restaurant_name": "KFC", "drop_amount": -162598, "share_pct": 65.0}
    ]
    legit_claim = "KFC represents the bulk of the decline, with a 65% share."
    assert unbacked_narrative_claims(legit_claim, evidence_with_share) == []
