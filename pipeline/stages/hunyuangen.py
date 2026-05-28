"""Hunyuan 3D generation stage: submit/poll/download via tencentcloud SDK."""

import base64
import time
from pathlib import Path

from pipeline.config import PipelineConfig
from pipeline.stages import register_stage
from pipeline.stages.base import BaseStage, StageError


def _load_tencentcloud():
    """Lazy-import tencentcloud SDK."""
    try:
        from tencentcloud.common.credential import Credential
        from tencentcloud.common.profile.client_profile import ClientProfile
        from tencentcloud.common.profile.http_profile import HttpProfile
        from tencentcloud.ai3d.v20250513 import ai3d_client, models
        return Credential, ClientProfile, HttpProfile, ai3d_client, models
    except ImportError:
        raise StageError(
            "tencentcloud-sdk-python not installed. "
            "Run: pip install pipeline-tool[hunyuan]"
        )


_POLL_INTERVAL = 10   # seconds
_MAX_WAIT = 1200       # 20 minutes timeout


@register_stage("hunyuangen")
class HunyuanGenStage(BaseStage):
    name = "hunyuangen"

    def run(self, config: PipelineConfig, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)

        Credential, ClientProfile, HttpProfile, ai3d_client, models = _load_tencentcloud()

        views_dir = Path(config.input.multi_views_dir)
        self.check_input_path(str(views_dir), "Multi-view images directory")

        # Encode multi-view images
        # 'front' is the default main view — used as ImageBase64, not in MultiViewImages
        multi_view_images = []
        front_b64 = None

        for view_name, filename in config.hunyuan.views.items():
            img_path = views_dir / filename
            self.check_input_path(str(img_path), f"View image ({view_name})")
            with open(img_path, "rb") as f:
                b64_data = base64.b64encode(f.read()).decode("utf-8")

            if view_name == "front":
                front_b64 = b64_data
            else:
                img = models.ViewImage()
                img.ViewType = view_name
                img.ViewImageBase64 = b64_data
                multi_view_images.append(img)

        if front_b64 is None:
            raise StageError("No 'front' view found in config — required as main image")

        # Create credential and client
        cred = Credential(config.hunyuan.secret_id, config.hunyuan.secret_key)
        http_profile = HttpProfile(endpoint="ai3d.tencentcloudapi.com")
        client_profile = ClientProfile(httpProfile=http_profile)
        client = ai3d_client.Ai3dClient(cred, config.hunyuan.region, client_profile)

        # Submit job
        req = models.SubmitHunyuanTo3DProJobRequest()
        req.Model = config.hunyuan.model
        req.ImageBase64 = front_b64
        req.MultiViewImages = multi_view_images
        req.GenerateType = "Normal"
        req.FaceCount = config.hunyuan.face_count
        req.EnablePBR = config.hunyuan.enable_pbr

        resp = client.SubmitHunyuanTo3DProJob(req)
        job_id = resp.JobId
        print(f"[hunyuangen] Job submitted: {job_id}")

        # Poll until complete
        elapsed = 0
        query_req = models.QueryHunyuanTo3DProJobRequest()
        query_req.JobId = job_id

        while elapsed < _MAX_WAIT:
            time.sleep(_POLL_INTERVAL)
            elapsed += _POLL_INTERVAL

            query_resp = client.QueryHunyuanTo3DProJob(query_req)
            status = query_resp.Status

            if status in ("SUCCESS", "DONE"):
                break
            elif status == "FAILED":
                raise StageError(
                    f"Hunyuan job {job_id} failed: {query_resp.ErrorMessage}"
                )

            print(f"[hunyuangen] Polling... status={status}, elapsed={elapsed}s")

        if elapsed >= _MAX_WAIT:
            raise StageError(f"Hunyuan job {job_id} timed out after {_MAX_WAIT}s")

        # Download result
        import urllib.request
        import zipfile
        import io

        result_files = getattr(query_resp, "ResultFile3Ds", None)
        if not result_files:
            raise StageError(
                f"Job {job_id} completed but no ResultFile3Ds in response"
            )

        # Prefer OBJ, fall back to first available
        dl_file = None
        for f in result_files:
            if f.Type == "obj":
                dl_file = f
                break
        if dl_file is None:
            dl_file = result_files[0]

        dl_path = output_dir / f"result.{dl_file.Type}"
        urllib.request.urlretrieve(dl_file.Url, str(dl_path))
        print(f"[hunyuangen] Downloaded {dl_file.Type} ({dl_path.stat().st_size} bytes)")

        # Check if result is a ZIP bundle — extract and find the OBJ
        if zipfile.is_zipfile(dl_path):
            with zipfile.ZipFile(dl_path) as zf:
                zf.extractall(output_dir)
            dl_path.unlink()  # remove the zip

            obj_files = list(output_dir.glob("*.obj"))
            if not obj_files:
                raise StageError("ZIP extracted but no .obj file found inside")

            # Rename first OBJ to obj.obj
            obj_files[0].rename(output_dir / "scale_obj.obj")
            print(f"[hunyuangen] Extracted OBJ + textures to {output_dir}")
        else:
            # Plain file — rename to obj.obj
            dl_path.rename(output_dir / "scale_obj.obj")

        print(f"[hunyuangen] Done: {output_dir}")
        return output_dir
