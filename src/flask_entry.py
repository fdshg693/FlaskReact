"""
Flaskアプリのエントリポイント
uv run {file_name} で起動

src直下に置かないと、VSCODEでのF5実行が動作しないため、必ずsrc直下に配置
（debug=trueにしていることで、2重プロセスが立ち上がることが原因のよう）
"""

from __future__ import annotations

# Import the Flask app factory without side effects
from server.flask_react_app import create_app


def main() -> None:
    app = create_app()
    app.run(host="0.0.0.0", port=8000, debug=True)


if __name__ == "__main__":
    main()
