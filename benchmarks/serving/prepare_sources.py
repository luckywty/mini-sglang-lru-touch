"""Freeze source copies and apply precisely the same observation entry hook."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import tarfile

ROOT = Path('/home/wangwei/ai-infra/Promotion')
REPO = ROOT / 'mini-sglang-lru-touch'
OUT = ROOT / 'benchmark-results'


def repair(original):
    result=original.replace('                finished = not req.can_decode',
                            '                finished = len(req.input_ids) >= req.max_device_len')
    marker='                if isinstance(req, ChunkedReq):\n                    continue'
    assert result.count(marker)==1
    result=result.replace(marker,marker+'\n                if req in self.finished_reqs:\n                    continue  # Discard already-released EOS/Abort results.')
    abort='            if req_to_free is not None:\n                self._free_req_resources(req_to_free)'
    assert result.count(abort)==1
    return result.replace(abort,abort+'\n                self.finished_reqs.add(req_to_free)  # In-flight host result still owns this object.')


def main():
    for label, source, sha in [
        ('H', ROOT/'mini-sglang-upstream', '9a91cfafe754aa85daee49998176275667eb58f2'),
        ('S', REPO, '7cd28c7bda8f6881558fe810694019d3e46994b5'),
    ]:
        target = ROOT / ('runtime-' + label)
        if not target.exists():
            target.mkdir()
            archive = subprocess.Popen(['git', 'archive', sha], cwd=source, stdout=subprocess.PIPE)
            subprocess.run(['tar', '-x', '-C', str(target)], stdin=archive.stdout, check=True)
            assert archive.wait() == 0
        # Idempotent: each source file is restored from the immutable Git commit.
        path = 'python/minisgl/scheduler/scheduler.py'
        original = subprocess.check_output(['git', 'show', sha+':'+path], cwd=source).decode()
        needle = '        super().__init__(config, self.engine.tp_cpu_group)'
        assert original.count(needle) == 1
        measured=repair(original).replace(needle, needle + '\n\n        from minisgl.benchmark.collector import install\n        install(self, config)')
        # Shared, separately identified correctness repair: overlap advances
        # device_len for the next forward before consuming this host result.
        # Termination must use the host sequence length after append_host.
        (target/path).write_text(measured)
        api_path='python/minisgl/server/api_server.py'
        api_original=subprocess.check_output(['git','show',sha+':'+api_path],cwd=source).decode()
        api_needle='uvicorn.run(app, host=host, port=port)'
        assert api_original.count(api_needle)==1
        api_measured=api_original.replace(api_needle,
                      'uvicorn.run(app, host=host, port=port, timeout_keep_alive=3600)')
        ack_header='state.stream_with_cancellation(state.stream_chat_completions(uid), request, uid),\n            media_type="text/event-stream",'
        assert api_measured.count(ack_header)==1
        api_measured=api_measured.replace(ack_header,ack_header+'\n            headers={"X-Bench-UID": str(uid)},')
        (target/api_path).write_text(api_measured)
        import difflib
        (OUT/'patches'/('http-keepalive-'+label+'.patch')).write_text(''.join(difflib.unified_diff(
                 api_original.splitlines(True),api_measured.splitlines(True),fromfile='a/'+api_path,tofile='b/'+api_path)))
        shutil.copyfile(REPO/'benchmarks/serving/collector.py', target/'python/minisgl/benchmark/collector.py')
        files = {str(p.relative_to(target)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted((target/'python').rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
        (OUT/'environment'/('runtime-'+label+'-files.json')).write_text(json.dumps(files, indent=2))
        frozen_manifest=json.dumps(files,indent=2).encode()
        fingerprint=hashlib.sha256(frozen_manifest).hexdigest()
        archives=OUT/'environment/sources'
        archives.mkdir(exist_ok=True)
        (archives/(fingerprint+'.json')).write_bytes(frozen_manifest)
        if not (archives/(fingerprint+'.tar.gz')).exists():
            with tarfile.open(archives/(fingerprint+'.tar.gz'),'w:gz') as archive:
                for name in files:
                    archive.add(target/name,arcname=name)
        patch = subprocess.run(['diff', '-u', '--label', 'a/'+path, '--label', 'b/'+path,
                                '/dev/stdin', str(target/path)], input=original, text=True, capture_output=True)
        (OUT/'patches'/('runtime-'+label+'.patch')).write_text(patch.stdout)
        correctness=repair(original)
        delta=subprocess.run(['diff','-u','--label','a/'+path,'--label','b/'+path,
                              '/dev/stdin',str(target/path)],input=correctness,text=True,capture_output=True)
        (OUT/'patches'/('measurement-only-'+label+'.patch')).write_text(delta.stdout)
        (OUT/'patches'/('correctness-'+label+'.patch')).write_text(''.join(difflib.unified_diff(
                 original.splitlines(True),correctness.splitlines(True),fromfile='a/'+path,tofile='b/'+path)))
    shutil.copyfile(REPO/'benchmarks/serving/collector.py', OUT/'patches/collector.py')


if __name__ == '__main__':
    main()
