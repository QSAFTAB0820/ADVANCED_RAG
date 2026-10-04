import io
import os
import tempfile
import unittest
from unittest.mock import patch

import agent
import main


class MissingKnowledgeBaseTests(unittest.TestCase):
    def test_startup_reports_missing_knowledge_base_without_internal_error(self):
        output = io.StringIO()

        with tempfile.TemporaryDirectory() as working_directory:
            original_directory = os.getcwd()
            try:
                os.chdir(working_directory)
                with patch("agent.ChatOpenAI"), patch(
                    "main.create_support_agent", side_effect=agent.create_support_agent
                ), patch("sys.stdout", output), self.assertRaises(SystemExit):
                    main.main()
            finally:
                os.chdir(original_directory)

        message = output.getvalue()
        self.assertIn("The support knowledge base is unavailable.", message)
        self.assertIn("Provide either 'knowledge_base.pdf' or 'knowledge_base.txt'", message)
        self.assertNotIn("FileNotFoundError", message)
        self.assertNotIn("Traceback", message)


if __name__ == "__main__":
    unittest.main()
