# Mflux-ComfyUI (v2)

mflux 0.10.x 기반으로 업그레이드된 ComfyUI 노드 모음입니다. 기존 그래프는 그대로 사용 가능하며(노드 재배선 불필요), 써드파티 모델, 확장된 퀀타 옵션, LoRA/ControlNet 개선, UI 내 MLX 버전 안내가 추가되었습니다.

- 백엔드: mflux 0.10.x만 지원 (레거시 0.4.1 런타임 미지원)
- 그래프 호환성: 레거시 입력을 내부에서 자동 변환하여 기존 그래프 유지
- OS/가속: macOS + MLX (Apple Silicon). MLX >= 0.27.0 권장

## 주요 기능
- 하나의 노드로 txt2img / img2img / LoRA / ControlNet 사용 (MFlux/Air → QuickMfluxNode)
- LoRA 파이프라인 + 유효성 검사(LoRA 사용 시 quantize=8 필요)
- ControlNet Canny 프리뷰 및 베스트‑에포트 컨디셔닝
- 써드파티 HF 리포 ID 지원(예: filipstrand/..., akx/...) + base_model 선택
- 퀀타 옵션: None, 3, 4, 5, 6, 8 (기본값 8)
- 메타데이터 저장(PNG + JSON, 레거시/신규 필드 동시 기록)

## 설치
- ComfyUI-Manager 권장: “Mflux-ComfyUI” 검색 후 설치

수동 설치:
1) cd /path/to/ComfyUI/custom_nodes
2) git clone https://github.com/joonsoome/Mflux-ComfyUI.git
3) ComfyUI venv 활성화 후 다음 설치
   - pip install --upgrade pip wheel setuptools
   - pip install 'mlx>=0.27.0' 'huggingface_hub>=0.24'
   - pip install 'mflux==0.10.0'
4) ComfyUI 재시작

비고:
- requirements.txt는 mflux==0.10.0 고정, pyproject는 mflux>=0.10.0 사용
- MLX < 0.27.0이면 UI에 권장 안내가 표시됩니다(업그레이드 권장)

## 제공 노드
- (MFlux/Air) QuickMfluxNode, Mflux Models Loader, Mflux Models Downloader, Mflux Custom Models
- (MFlux/Pro) Mflux Img2Img, Mflux Loras Loader, Mflux ControlNet Loader

## 개발자 노트 — 추가 노드 및 테스트

이 포크에는 업스트림 빠른 시작에는 없는 몇 가지 Pro 단계 노드와 단위 테스트가 포함되어 있습니다. 기여자 및 고급 사용자를 위한 내용입니다:

- `MfluxUpscale` (MFlux/Pro): ComfyUI `IMAGE` 텐서를 우선 입력으로 받는 업스케일 노드입니다. 레거시 파일 선택 위젯에 대한 하위 호환 지원도 포함되어 있으며, IMAGE 입력을 받을 경우 임시 PNG를 생성합니다. 기본적으로 PNG와 JSON 메타데이터를 저장합니다.
- `MfluxFill`, `MfluxDepth`, `MfluxRedux` (MFlux/Pro): 인페인트/필, 깊이 조건 생성, 이미지 변주용 Flux 도구입니다. 이 노드들은 `masked_image_path`, `depth_image_path`, `redux_image_paths`, `redux_image_strengths` 같은 phase-2 스타일의 설정 키를 핵심 `generate_image` 흐름으로 전달합니다.

`tests/` 폴더에 추가된 테스트는 다음을 포함합니다:

- 메타데이터 저장 및 파라미터 마이그레이션을 확인하는 테스트 (`tests/test_metadata_contents.py`).
- `save_images_with_metadata` 호출을 스파이하여 Fill/Depth/Redux 노드 동작을 단위검증하는 테스트 (`tests/test_fill_node.py`, `tests/test_depth_node.py`, `tests/test_redux_node.py`).
- `generate_image`의 TypeError 폴백 경로(알 수 없는 키 제거 후 재시도)를 검증하는 테스트 (`tests/test_config_fallback.py`).
- ComfyUI IMAGE 텐서 변환 및 하위 호환성 보장을 확인하는 업스케일 관련 내부 테스트 (`tests/test_upscale_internals.py`).

Kontext, In-Context LoRA, CatVTON, Concept-Attention 같은 Flux 확장을 계획한다면, 기존 테스트 패턴과 `Mflux_Comfy/Mflux_Core.py`의 TypeError 재시도 로직이 API 변화에 대응하는 좋은 출발점이 됩니다.

## 사용 팁
- LoRA + quantize < 8 미지원 → LoRA 사용 시 quantize=8로 설정
- 가로/세로는 8의 배수 권장
- dev 모델만 guidance가 반영되며, schnell은 무시(설정해도 무방)
- 시드 -1은 매번 랜덤. 크기/품질 프리셋 제공

### 경로
- 양자화 모델: ComfyUI/models/Mflux
- LoRA: ComfyUI/models/loras (정리를 위해 models/loras/Mflux 폴더 권장)
- 풀 모델 캐시(HF): ~/Library/Caches/mflux (혹은 시스템 기본 캐시)

## 워크플로우
workflows 폴더에 예제가 포함되어 있습니다(txt2img, img2img, LoRA 스택, ControlNet canny 등). 노드가 빨간색이면 ComfyUI-Manager의 “One‑click Install Missing Nodes” 기능을 사용하세요.

## 감사
- mflux by @filipstrand 및 컨트리뷰터: https://github.com/filipstrand/mflux
- 일부 구조는 @CharafChnioune의 MFLUX-WEBUI(Apache‑2.0)에서 영감을 받았으며, 참조된 부분에 라이선스 주석을 포함했습니다.

## 라이선스
MIT
