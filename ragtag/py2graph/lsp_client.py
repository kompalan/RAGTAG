import subprocess
import threading
from pathlib import Path
import logging

import pylspclient
import pylspclient.lsp_pydantic_strcuts as lsp_structs


logger = logging.getLogger(__name__)


class _StderrReader(threading.Thread):
    def __init__(self, pipe):
        super().__init__(daemon=True)
        self.pipe = pipe

    def run(self):
        for line in iter(self.pipe.readline, b""):
            # Pyright writes useful logs to stderr
            logger.error(f'LSP Error: {line.decode("utf-8").rstrip()}')


class _StdoutReader(threading.Thread):
    def __init__(self, pipe):
        super().__init__(daemon=True)
        self.pipe = pipe

    def run(self):
        for line in iter(self.pipe.readline, b""):
            # Pyright writes useful logs to stderr
            logger.info(f'LSP Info: {line.decode("utf-8").rstrip()}')


class PyrightLsp:
    def __init__(self, workspace_root: str):
        self.workspace_root = str(Path(workspace_root).resolve())
        self.proc = None
        self.lsp_client = None

        self._open_versions = {}  # uri -> version

    def window(self, logMessage: str):
        logger.info(f"From LSP Client: {logMessage}")

    def start(self):
        self.proc = subprocess.Popen(
            ["python", "-m", "pylsp"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert self.proc.stdin and self.proc.stdout and self.proc.stderr

        _StderrReader(self.proc.stderr).start()
        # _StdoutReader(self.proc.stdout).start()

        json_rpc = pylspclient.JsonRpcEndpoint(self.proc.stdin, self.proc.stdout)
        lsp = pylspclient.LspEndpoint(json_rpc, {}, {"window": self.window}, 10)

        # lsp = pylspclient.LspEndpoint(json_rpc)
        self.lsp_client = pylspclient.LspClient(lsp)

        root_uri = Path(self.workspace_root).as_uri()
        caps = {
            "textDocument": {
                "definition": {"dynamicRegistration": False},
                "references": {"dynamicRegistration": False},
                "synchronization": {"dynamicRegistration": False},
            },
            "workspace": {"workspaceFolders": True},
        }

        # caps = {
        #     "textDocument": {
        #         "completion": {
        #             "completionItem": {
        #                 "commitCharactersSupport": True,
        #                 "documentationFormat": ["markdown", "plaintext"],
        #                 "snippetSupport": True,
        #             }
        #         }
        #     }
        # }
        #
        print(
            self.lsp_client.initialize(
                # processId=self.proc.pid,
                processId=self.proc.pid,
                rootUri=root_uri,
                rootPath=None,
                capabilities=caps,
                workspaceFolders=[
                    {"name": Path(self.workspace_root).name, "uri": root_uri}
                ],
                trace="verbose",
                initializationOptions=None,
            )
        )
        self.lsp_client.initialized()
        # lsp.send_message("$/setTrace", value="verbose")

    def stop(self):
        if not self.lsp_client:
            return
        try:
            self.lsp_client.shutdown()
            self.lsp_client.exit()
        finally:
            if self.proc:
                self.proc.terminate()

    def _uri(self, path: str) -> str:
        return Path(path).resolve().as_uri()

    def open_file(self, path: str):
        assert self.lsp_client
        uri = self._uri(path)
        text = Path(path).read_text(encoding="utf-8")
        version = 1
        self._open_versions[uri] = version

        self.lsp_client.didOpen(
            lsp_structs.TextDocumentItem(
                uri=uri,
                languageId=lsp_structs.LanguageIdentifier.PYTHON,
                version=version,
                text=text,
            )
        )
        return uri

    def change_file_full(self, path: str, new_text: str):
        """Fast + simple: send whole-file replacement. Good enough for analysis loops."""
        assert self.lsp_client
        uri = self._uri(path)
        version = self._open_versions.get(uri, 0) + 1
        self._open_versions[uri] = version

        doc = lsp_structs.VersionedTextDocumentIdentifier(uri=uri, version=version)
        change = lsp_structs.TextDocumentContentChangeEvent(text=new_text)
        self.lsp_client.didChange(
            lsp_structs.DidChangeTextDocumentParams(doc, [change])
        )

    def definition(self, path: str, line0: int, col0: int):
        """0-based line/col. Put (line0,col0) on the `foo` in `super().foo()`."""
        assert self.lsp_client
        uri = self._uri(path)
        return self.lsp_client.definition(
            lsp_structs.TextDocumentIdentifier(uri=uri),
            lsp_structs.Position(line=line0, character=col0),
        )
