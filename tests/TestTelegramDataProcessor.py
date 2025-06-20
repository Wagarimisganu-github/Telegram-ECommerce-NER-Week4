import unittest
import pandas as pd
from io import StringIO
from your_module_name import TelegramDataProcessor  # Import your actual module name

class TestTelegramDataProcessor(unittest.TestCase):

    # Setup before each test case
    def setUp(self):
        # Sample CSV data to use in testing
        sample_data = StringIO("""Message
        ብስራተ ገብርኤል ብር 500 ነው እና ገርጂ 4ኪሎ ይሄው
        Phone number is +251911234567
        No price or location info here
        """)
        # Initialize the TelegramDataProcessor with the sample data
        self.processor = TelegramDataProcessor(sample_data)

    def test_check_and_remove_nan(self):
        # Add NaN value to the dataframe for testing
        self.processor.df.loc[3] = [None]
        self.processor.check_and_remove_nan('Message')
        # Ensure the NaN row was removed
        self.assertEqual(self.processor.df.shape[0], 3)

    def test_remove_emojis(self):
        message_with_emoji = "Hello 😊, ብር 500!"
        cleaned_message = self.processor.remove_emojis(message_with_emoji)
        # Ensure emojis are removed
        self.assertEqual(cleaned_message, "Hello , ብር 500!")

    def test_label_message(self):
        # Test a message that should be labeled
        message = "ብስራተ ገብርኤል ብር 500"
        labeled_message = self.processor.label_message(message)
        expected_output = "ብስራተ ገብርኤል I-LOC\nብር I-PRICE\n500 I-PRICE"
        self.assertEqual(labeled_message, expected_output)

    def test_is_amharic(self):
        # Test a message with Amharic text
        amharic_message = "ብስራተ ገብርኤል"
        self.assertTrue(self.processor.is_amharic(amharic_message))

        # Test a message without Amharic text
        english_message = "Hello world"
        self.assertFalse(self.processor.is_amharic(english_message))

    def test_classify_message(self):
        # Test classification of a message containing location and price
        message = "ብስራተ ገብርኤል ብር 500"
        category = self.processor.classify_message(message)
        self.assertEqual(category, 'price')

        # Test an uncategorized message
        uncategorized_message = "No relevant info"
        category = self.processor.classify_message(uncategorized_message)
        self.assertEqual(category, 'uncategorized')

    def test_clean_messages(self):
        # Test if emoji removal is applied correctly
        self.processor.df['Message'] = ["Hello 😊", "ብር 500"]
        self.processor.clean_messages()
        cleaned_message = self.processor.df['Message'][0]
        # Ensure emojis are removed
        self.assertEqual(cleaned_message, "Hello ")

if __name__ == '__main__':
    unittest.main()
