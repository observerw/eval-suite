from eval_suite.utils.extract import extract_boxed, extract_code


class TestExtractCode:
    def test_extract_code_with_simple_codeblock(self):
        """Test extracting a simple code block."""
        text = """
        Here is a code example:
        ```python
        def hello():
            print("Hello, world!")
        ```
        """
        results = extract_code(text)
        python_results = results.get("python")

        assert len(python_results) == 1
        assert python_results[0].lang == "python"
        assert python_results[0].code == 'def hello():\n    print("Hello, world!")'
        assert python_results[0].group == "default"
        assert python_results[0].id is None

    def test_extract_code_with_group_and_id(self):
        """Test extracting code with group and id specified."""
        text = """
        ```python [solution] {problem1}
        def solve(n):
            return n * 2
        ```
        """
        results = extract_code(text)
        solution = results.get("python", group="solution", id="problem1")

        assert solution is not None
        assert solution.lang == "python"
        assert solution.group == "solution"
        assert solution.id == "problem1"
        assert solution.code == "def solve(n):\n    return n * 2"

    def test_extract_code_with_multiple_codeblocks(self):
        """Test extracting multiple code blocks with different languages."""
        text = """
        Python example:
        ```python
        x = 10
        ```
        
        JavaScript example:
        ```javascript
        let x = 10;
        ```
        
        Another Python example:
        ```python [test]
        assert x == 10
        ```
        """
        results = extract_code(text)

        python_results = results.get("python")
        assert len(python_results) == 2

        js_results = results.get("javascript")
        assert len(js_results) == 1
        assert js_results[0].code == "let x = 10;"

        test_results = results.get("python", group="test")
        assert len(test_results) == 1
        assert test_results[0].code == "assert x == 10"

    def test_extract_code_with_no_codeblocks(self):
        """Test extracting code from text with no code blocks."""
        text = "This is plain text with no code blocks."
        results = extract_code(text)

        assert len(results.root) == 0


class TestExtractBoxed:
    def test_extract_boxed_simple(self):
        """Test extracting content from \\boxed{} notation."""
        text = "The answer is \\boxed{42}"
        result = extract_boxed(text)

        assert result == "42"

    def test_extract_boxed_with_multiline(self):
        """Test extracting multiline content from \\boxed{} notation."""
        text = """
        The solution is \\boxed{
            x = 10
            y = 20
            z = x + y
        }
        """
        result = extract_boxed(text)

        assert result == "x = 10\n            y = 20\n            z = x + y"

    def test_extract_boxed_not_found(self):
        """Test that None is returned when no boxed content is found."""
        text = "There is no boxed content here."
        result = extract_boxed(text)

        assert result is None
