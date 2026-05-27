# Makes this folder a Python package.
# Inside src/, always use RELATIVE imports:  from .rag_chain import ...
# From the project root, you can use:        from src.rag_chain import ...

"""Why it has no code:
In a well-structured project, you don't put logic inside __init__.py. Each module (api.py, rag_chain.py, etc.) is responsible for its own code. Putting imports or logic in __init__.py can cause circular import problems — for example, if api.py imports from __init__.py which imports from api.py, Python gets stuck in a loop.
The industry standard (which this follows) is:

Small projects → __init__.py is empty or has only a comment
Large libraries (like LangChain itself) → __init__.py re-exports public APIs for cleaner imports

For your agent, keeping it as just a comment is exactly right. No changes needed there."""