import unittest
from src.agent.numeric_grounding import ungrounded_numbers, unbacked_narrative_claims

class TestB2Grounding(unittest.TestCase):
    def setUp(self):
        # 15 rows from R2
        self.r2_evidence = [
            {'restaurant_name': 'KFC', 'total_sales_amount': 42466314.0, 'total_order_count': 61731},
            {'restaurant_name': "Domino's Pizza", 'total_sales_amount': 35671213.0, 'total_order_count': 53547},
            {'restaurant_name': 'Pizza Hut', 'total_sales_amount': 32128653.0, 'total_order_count': 53541},
            {'restaurant_name': 'Behrouz Biryani', 'total_sales_amount': 25922775.0, 'total_order_count': 30730},
            {'restaurant_name': 'Subway', 'total_sales_amount': 25483225.0, 'total_order_count': 44866},
            {'restaurant_name': 'Oven Story Pizza', 'total_sales_amount': 23491566.0, 'total_order_count': 34756},
            {'restaurant_name': "McDonald's", 'total_sales_amount': 20492809.0, 'total_order_count': 31607},
            {'restaurant_name': 'Barbeque Nation', 'total_sales_amount': 14352127.0, 'total_order_count': 14220},
            {'restaurant_name': 'Faasos - Wraps & Rolls', 'total_sales_amount': 14060026.0, 'total_order_count': 35956},
            {'restaurant_name': 'Biryani By Kilo', 'total_sales_amount': 13994523.0, 'total_order_count': 13425},
            {'restaurant_name': 'Kwality Walls Frozen Dessert and Ice Cream Shop', 'total_sales_amount': 12070739.0, 'total_order_count': 29791},
            {'restaurant_name': 'The Good Bowl', 'total_sales_amount': 11995766.0, 'total_order_count': 19417},
            {'restaurant_name': 'Dindigul Thalappakatti', 'total_sales_amount': 10614598.0, 'total_order_count': 11965},
            {'restaurant_name': 'Wow! Momo', 'total_sales_amount': 10559658.0, 'total_order_count': 21906},
            {'restaurant_name': 'Sweet Truth - Cake and Desserts', 'total_sales_amount': 10413158.0, 'total_order_count': 14366}
        ]

    def test_exact_r2_answer_passes(self):
        """The exact R2 answer with 1..15 numbered list, rupee signs, and row counts must pass grounding."""
        r2_text = (
            "The top 15 restaurants by total sales and order count are:\n"
            "1. **KFC** – ₹42,466,314 in sales from 61,731 orders.\n"
            "2. **Domino's Pizza** – ₹35,671,213 in sales from 53,547 orders.\n"
            "3. **Pizza Hut** – ₹32,128,653 in sales from 53,541 orders.\n"
            "4. **Behrouz Biryani** – ₹25,922,775 in sales from 30,730 orders.\n"
            "5. **Subway** – ₹25,483,225 in sales from 44,866 orders.\n"
            "6. **Oven Story Pizza** – ₹23,491,566 in sales from 34,756 orders.\n"
            "7. **McDonald's** – ₹20,492,809 in sales from 31,607 orders.\n"
            "8. **Barbeque Nation** – ₹14,352,127 in sales from 14,220 orders.\n"
            "9. **Faasos - Wraps & Rolls** – ₹14,060,026 in sales from 35,956 orders.\n"
            "10. **Biryani By Kilo** – ₹13,994,523 in sales from 13,425 orders.\n"
            "11. **Kwality Walls Frozen Dessert and Ice Cream Shop** – ₹12,070,739 in sales from 29,791 orders.\n"
            "12. **The Good Bowl** – ₹11,995,766 in sales from 19,417 orders.\n"
            "13. **Dindigul Thalappakatti** – ₹10,614,598 in sales from 11,965 orders.\n"
            "14. **Wow! Momo** – ₹10,559,658 in sales from 21,906 orders.\n"
            "15. **Sweet Truth - Cake and Desserts** – ₹10,413,158 in sales from 14,366 orders.\n"
        )
        ungrounded = ungrounded_numbers(r2_text, self.r2_evidence)
        self.assertEqual(ungrounded, [], f"Expected 0 ungrounded numbers, got {ungrounded}")

    def test_fabricated_unscaled_number_fails(self):
        """Fabricated '14,000,000' must fail even though 13,994,523 exists in evidence."""
        fabricated_text = "Biryani By Kilo had 14,000,000 in sales."
        ungrounded = ungrounded_numbers(fabricated_text, self.r2_evidence)
        self.assertIn("14,000,000", ungrounded)

    def test_scaled_suffix_14m_matches_evidence(self):
        """'14M' or '14 million' is accepted because 13,994,523 rounds to 14.0M."""
        scaled_text = "Biryani By Kilo achieved approximately 14M in sales."
        ungrounded = ungrounded_numbers(scaled_text, self.r2_evidence)
        self.assertEqual(ungrounded, [])

    def test_arabic_scaled_and_indic_numbers(self):
        """Arabic scaled suffixes like '١٤ مليون' match rounded figures."""
        arabic_text = "حققت برياني باي كيلو ما يقارب ١٤ مليون مبيعات."
        ungrounded = ungrounded_numbers(arabic_text, self.r2_evidence)
        self.assertEqual(ungrounded, [])

    def test_unbacked_narrative_claim_detection(self):
        """Words like 'represents the bulk' without >50% share proof must be flagged."""
        text = "KFC represents the bulk of the overall revenue."
        violations = unbacked_narrative_claims(text, self.r2_evidence)
        self.assertTrue(len(violations) > 0)
        self.assertIn("bulk", violations[0].lower())


if __name__ == "__main__":
    unittest.main()
