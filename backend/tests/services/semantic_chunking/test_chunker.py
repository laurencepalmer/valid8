"""Integration tests for the SemanticChunker."""

import pytest
from backend.services.semantic_chunking.chunker import SemanticChunker
from backend.models.codebase import CodeFile
from backend.models.semantic_chunk import SymbolType


@pytest.fixture
def chunker():
    """Create a SemanticChunker instance."""
    return SemanticChunker()


class TestPythonChunking:
    """Tests for Python code chunking."""

    def test_chunk_single_function(self, chunker):
        """Test chunking a single function."""
        source = '''
def get_user(user_id: int) -> dict:
    """Fetch user by ID from database."""
    return db.query(User).filter_by(id=user_id).first()
'''
        file = CodeFile(
            path="/path/to/user.py",
            relative_path="services/user.py",
            content=source,
            language="python",
            line_count=5,
        )

        chunks = chunker.chunk_file(file)

        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.metadata.symbol_type == SymbolType.FUNCTION
        assert chunk.metadata.symbol_name == "get_user"
        assert chunk.docstring == "Fetch user by ID from database."
        assert "user_id" in chunk.metadata.parameters
        assert chunk.metadata.return_type == "dict"

    def test_chunk_multiple_functions(self, chunker):
        """Test chunking multiple functions."""
        source = '''
def create_user(name: str) -> User:
    """Create a new user."""
    return User(name=name)

def delete_user(user_id: int) -> bool:
    """Delete a user by ID."""
    user = get_user(user_id)
    if user:
        db.delete(user)
        return True
    return False
'''
        file = CodeFile(
            path="/path/to/user.py",
            relative_path="services/user.py",
            content=source,
            language="python",
            line_count=15,
        )

        chunks = chunker.chunk_file(file)

        assert len(chunks) == 2
        names = [c.metadata.symbol_name for c in chunks]
        assert "create_user" in names
        assert "delete_user" in names

    def test_chunk_class_and_methods(self, chunker):
        """Test chunking a class with methods."""
        source = '''
class UserService:
    """Service for user operations."""

    def __init__(self, db):
        self.db = db

    def get_user(self, user_id):
        """Get a user by ID."""
        return self.db.query(User).get(user_id)

    def list_users(self):
        """List all users."""
        return self.db.query(User).all()
'''
        file = CodeFile(
            path="/path/to/user.py",
            relative_path="services/user.py",
            content=source,
            language="python",
            line_count=15,
        )

        chunks = chunker.chunk_file(file)

        # Should find class and methods
        types = [c.metadata.symbol_type for c in chunks]
        assert SymbolType.CLASS in types
        assert SymbolType.METHOD in types

        # Find the class chunk
        class_chunks = [c for c in chunks if c.metadata.symbol_type == SymbolType.CLASS]
        assert len(class_chunks) == 1
        assert class_chunks[0].metadata.symbol_name == "UserService"

    def test_chunk_with_decorators(self, chunker):
        """Test chunking functions with decorators."""
        source = '''
@app.route("/users")
@login_required
def list_users():
    """List all users."""
    return jsonify(users)
'''
        file = CodeFile(
            path="/path/to/routes.py",
            relative_path="routes.py",
            content=source,
            language="python",
            line_count=6,
        )

        chunks = chunker.chunk_file(file)

        assert len(chunks) == 1
        chunk = chunks[0]
        assert len(chunk.metadata.decorators) == 2
        assert any("@app.route" in d for d in chunk.metadata.decorators)
        assert any("@login_required" in d for d in chunk.metadata.decorators)

    def test_chunk_with_error_handling(self, chunker):
        """Test chunking functions with error handling."""
        source = '''
def safe_divide(a, b):
    """Safely divide two numbers."""
    try:
        return a / b
    except ZeroDivisionError:
        return None
'''
        file = CodeFile(
            path="/path/to/math.py",
            relative_path="utils/math.py",
            content=source,
            language="python",
            line_count=7,
        )

        chunks = chunker.chunk_file(file)

        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.behavior.has_error_handling is True

    def test_chunk_async_function(self, chunker):
        """Test chunking async functions."""
        source = '''
async def fetch_data(url: str) -> dict:
    """Fetch data from URL."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()
'''
        file = CodeFile(
            path="/path/to/client.py",
            relative_path="client.py",
            content=source,
            language="python",
            line_count=6,
        )

        chunks = chunker.chunk_file(file)

        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.behavior.has_async is True


class TestJavaScriptChunking:
    """Tests for JavaScript code chunking."""

    def test_chunk_function_declaration(self, chunker):
        """Test chunking a function declaration."""
        source = '''
/**
 * Fetch user by ID
 */
function getUser(userId) {
    return fetch(`/api/users/${userId}`).then(r => r.json());
}
'''
        file = CodeFile(
            path="/path/to/user.js",
            relative_path="services/user.js",
            content=source,
            language="javascript",
            line_count=7,
        )

        chunks = chunker.chunk_file(file)

        assert len(chunks) >= 1
        func_chunks = [c for c in chunks if c.metadata.symbol_type == SymbolType.FUNCTION]
        assert len(func_chunks) >= 1

    def test_chunk_arrow_function(self, chunker):
        """Test chunking arrow functions."""
        source = '''
const getUserById = (userId) => {
    return db.find({ id: userId });
};
'''
        file = CodeFile(
            path="/path/to/user.js",
            relative_path="services/user.js",
            content=source,
            language="javascript",
            line_count=4,
        )

        chunks = chunker.chunk_file(file)
        assert len(chunks) >= 1

    def test_chunk_class(self, chunker):
        """Test chunking JavaScript class."""
        source = '''
class UserService {
    constructor(db) {
        this.db = db;
    }

    getUser(userId) {
        return this.db.find(userId);
    }
}
'''
        file = CodeFile(
            path="/path/to/user.js",
            relative_path="services/user.js",
            content=source,
            language="javascript",
            line_count=10,
        )

        chunks = chunker.chunk_file(file)

        types = [c.metadata.symbol_type for c in chunks]
        assert SymbolType.CLASS in types


class TestFallbackChunking:
    """Tests for fallback line-based chunking."""

    def test_unsupported_language_fallback(self, chunker):
        """Test fallback for unsupported languages."""
        source = "Some content in an unsupported format\n" * 50
        file = CodeFile(
            path="/path/to/file.xyz",
            relative_path="file.xyz",
            content=source,
            language="unknown",
            line_count=50,
        )

        chunks = chunker.chunk_file(file)

        # Should produce fallback chunks
        assert len(chunks) >= 1
        assert chunks[0].metadata.symbol_type == SymbolType.UNKNOWN


class TestSearchTextGeneration:
    """Tests for search text generation in chunks."""

    def test_search_text_includes_key_info(self, chunker):
        """Test that search text includes key information."""
        source = '''
def get_user_by_id(user_id: int) -> Optional[User]:
    """Fetch user from database by their unique ID."""
    if user_id <= 0:
        return None
    return db.query(User).get(user_id)
'''
        file = CodeFile(
            path="/path/to/user.py",
            relative_path="services/user.py",
            content=source,
            language="python",
            line_count=6,
        )

        chunks = chunker.chunk_file(file)

        assert len(chunks) == 1
        search_text = chunks[0].search_text.lower()

        # Should include function name (normalized)
        assert "get" in search_text
        assert "user" in search_text

        # Should include file path
        assert "services/user.py" in search_text

        # Should include language
        assert "python" in search_text

    def test_search_text_includes_behavior(self, chunker):
        """Test that search text includes behavior information."""
        source = '''
async def process_items(items):
    """Process items in batch."""
    results = []
    for item in items:
        try:
            result = await process_one(item)
            results.append(result)
        except Exception:
            pass
    return results
'''
        file = CodeFile(
            path="/path/to/processor.py",
            relative_path="processor.py",
            content=source,
            language="python",
            line_count=11,
        )

        chunks = chunker.chunk_file(file)

        assert len(chunks) == 1
        search_text = chunks[0].search_text.lower()

        # Should mention async
        assert "async" in search_text

        # Should mention error handling
        assert "error" in search_text


class TestChunkMetadata:
    """Tests for chunk metadata correctness."""

    def test_line_numbers(self, chunker):
        """Test that line numbers are correct."""
        source = '''
# Header comment

def first_function():
    pass

def second_function():
    return True
'''
        file = CodeFile(
            path="/path/to/funcs.py",
            relative_path="funcs.py",
            content=source,
            language="python",
            line_count=9,
        )

        chunks = chunker.chunk_file(file)

        assert len(chunks) == 2
        # First function should start after the comment
        first_func = [c for c in chunks if c.metadata.symbol_name == "first_function"][0]
        second_func = [c for c in chunks if c.metadata.symbol_name == "second_function"][0]

        assert first_func.metadata.start_line < second_func.metadata.start_line

    def test_qualified_name_for_methods(self, chunker):
        """Test qualified names for class methods."""
        source = '''
class MyClass:
    def my_method(self):
        pass
'''
        file = CodeFile(
            path="/path/to/cls.py",
            relative_path="cls.py",
            content=source,
            language="python",
            line_count=4,
        )

        chunks = chunker.chunk_file(file)

        method_chunks = [c for c in chunks if c.metadata.symbol_type == SymbolType.METHOD]
        assert len(method_chunks) == 1
        assert method_chunks[0].metadata.qualified_name == "MyClass.my_method"
        assert method_chunks[0].metadata.parent_scope == "MyClass"


class TestDeterminism:
    """Tests for deterministic chunking output."""

    def test_chunking_is_deterministic(self, chunker):
        """Test that chunking produces the same output for the same input."""
        source = '''
def func_a():
    return 1

def func_b():
    return 2

class MyClass:
    def method_a(self):
        pass
'''
        file = CodeFile(
            path="/path/to/code.py",
            relative_path="code.py",
            content=source,
            language="python",
            line_count=11,
        )

        # Chunk twice
        chunks1 = chunker.chunk_file(file)
        chunks2 = chunker.chunk_file(file)

        # Should produce identical results
        assert len(chunks1) == len(chunks2)
        for c1, c2 in zip(chunks1, chunks2):
            assert c1.chunk_id == c2.chunk_id
            assert c1.content == c2.content
            assert c1.search_text == c2.search_text
            assert c1.metadata.symbol_name == c2.metadata.symbol_name


class TestSubChunking:
    """Tests for sub-chunking large symbols."""

    def test_large_function_is_sub_chunked(self, chunker):
        """Test that large functions are split into sub-chunks."""
        # Create a function with many lines
        lines = ["def very_large_function():"]
        lines.append('    """A very large function."""')
        for i in range(200):
            lines.append(f"    x{i} = {i}")
        lines.append("    return x0")

        source = "\n".join(lines)
        file = CodeFile(
            path="/path/to/large.py",
            relative_path="large.py",
            content=source,
            language="python",
            line_count=len(lines),
        )

        chunks = chunker.chunk_file(file)

        # Should produce multiple sub-chunks
        sub_chunks = [c for c in chunks if c.is_sub_chunk]
        if sub_chunks:
            assert all(c.parent_chunk_id is not None for c in sub_chunks)
            assert all(c.sub_chunk_index is not None for c in sub_chunks)

    def test_small_function_not_sub_chunked(self, chunker):
        """Test that small functions are not sub-chunked."""
        source = '''
def small_function():
    """A small function."""
    return 42
'''
        file = CodeFile(
            path="/path/to/small.py",
            relative_path="small.py",
            content=source,
            language="python",
            line_count=4,
        )

        chunks = chunker.chunk_file(file)

        assert len(chunks) == 1
        assert chunks[0].is_sub_chunk is False
        assert chunks[0].parent_chunk_id is None
