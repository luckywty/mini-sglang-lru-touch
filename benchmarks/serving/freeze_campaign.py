"""Freeze final source/tool/model identities without overwriting pilot candidates."""
from pathlib import Path
import hashlib
import importlib.metadata as metadata
import json
import shutil
import subprocess
import tarfile
import time

from run_experiments import OUT,ROOT,MODEL,atomic,sha


def main():
    target=OUT/'frozen-config.json'
    config=json.loads(target.read_text())
    assert config.get('W5_stability_verified')
    calibration=OUT/'preflight/fixed-arrival-calibration/acceptance.json'
    cal=json.loads(calibration.read_text())
    old=target.read_bytes();old_hash=hashlib.sha256(old).hexdigest()
    archive=OUT/'environment/config-history';archive.mkdir(exist_ok=True)
    (archive/(old_hash+'.json')).write_bytes(old)
    model_manifest=json.loads((OUT/'environment/model-files.json').read_text())
    for name,record in model_manifest.items():
        path=Path(MODEL)/name
        assert path.stat().st_size==record['bytes'] and sha(path)==record['sha256'],f'Model file changed: {path}'
    repo=ROOT/'mini-sglang-lru-touch'
    files={str(p.relative_to(repo)):sha(p) for p in sorted((repo/'benchmarks/serving').iterdir()) if p.is_file()}
    data=json.dumps(files,indent=2).encode();tool_hash=hashlib.sha256(data).hexdigest()
    tools=OUT/'environment/tooling';tools.mkdir(exist_ok=True)
    (tools/(tool_hash+'.json')).write_bytes(data)
    if not (tools/(tool_hash+'.tar.gz')).exists():
        with tarfile.open(tools/(tool_hash+'.tar.gz'),'w:gz') as tar:
            for name in files:tar.add(repo/name,arcname=name)
    shutil.copyfile(repo/'uv.lock',OUT/'environment/uv.lock')
    versions={d.metadata['Name']:d.version for d in metadata.distributions()}
    atomic(OUT/'environment/final-package-versions.json',versions)
    config.update(measurement_frozen=True,measurement_frozen_wall=time.time(),
        collector_mode='light',tail_calibration_passed=bool(cal['passed']),
        calibration_sha256=sha(calibration),model_manifest_sha256=sha(OUT/'environment/model-files.json'),
        model_revision='local-sha256:'+model_manifest['model.safetensors']['sha256'],
        tokenizer_hash=model_manifest['tokenizer.json']['sha256'],
        runtime_H_manifest_sha256=sha(OUT/'environment/runtime-H-files.json'),
        runtime_S_manifest_sha256=sha(OUT/'environment/runtime-S-files.json'),tooling_sha256=tool_hash,
        dtype='bfloat16',attention_backend='fi/fa2',page_size=1,kv_tokens=16384,
        context_limit=2304,max_running_requests=16,normal_prefill_tokens=256,
        expected_cuda_graph_sizes=[1,2,4,8,16],http_keepalive_seconds=3600,
        quality_retries=2,quality_retry_reasons=['client lag exceeds frozen limit','60s thermal range exceeds2C'],
        quality_rule='Keep every attempt; select the first quality-valid restart per seed, up to2 replacements. Slow performance or timeouts alone are not exclusion grounds.',
        evidence_scope=('fixed-length timing with stated upstream numerical-equivalence limitation' if cal['passed'] else
                        'Tail calibration did not meet preregistered tolerance: ITL effects exploratory, excluded from resume gains. Retain absolute observations, paired CIs, throughput/failure/fairness evidence and every failed gate.'))
    atomic(target,config)
    atomic(OUT/'environment/frozen-tooling-commit.json',dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
                   worktree=subprocess.check_output(['git','status','--short'],cwd=repo,text=True),tooling_sha256=tool_hash))
    print(json.dumps({'frozen_config_sha256':sha(target),'tail_calibration_passed':cal['passed'],
                      'tooling_sha256':tool_hash,'evidence_scope':config['evidence_scope']}),flush=True)


if __name__=='__main__':main()
