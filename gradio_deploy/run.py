# This is a file to show demo of deployment for the Enhance_INR_with_TL project
# It includes code for exporting the model and running inference on a sample input using gradio.

import os
import numpy as np
import gradio as gr
from pathlib import Path


TARGET_DIR = Path("/local_dataset/Chest_CT")
INPUT_IMAGES = [
    TARGET_DIR / "046.png",
    TARGET_DIR / "092.png",
    TARGET_DIR / "059.png",
    TARGET_DIR / "112.png",
    TARGET_DIR / "100.png",
    TARGET_DIR / "001.png",
    TARGET_DIR / "033.png",
    TARGET_DIR / "070.png",
]

RESULTS_ROOT = Path("/home/choah76/workspace2/Enhance_INR_with_TL/gradio_deploy/results/siren/Chest_CT")
NPY_PATHS = [
    (RESULTS_ROOT / "5/imid[46]_intermediate_imgs[uint8].npy" , RESULTS_ROOT / "scratch/imid[46]_intermediate_imgs[uint8].npy" ),
    (RESULTS_ROOT / "5/imid[92]_intermediate_imgs[uint8].npy" , RESULTS_ROOT / "scratch/imid[92]_intermediate_imgs[uint8].npy" ),
    (RESULTS_ROOT / "2/imid[59]_intermediate_imgs[uint8].npy" , RESULTS_ROOT / "scratch/imid[59]_intermediate_imgs[uint8].npy" ),
    (RESULTS_ROOT / "2/imid[112]_intermediate_imgs[uint8].npy", RESULTS_ROOT / "scratch/imid[112]_intermediate_imgs[uint8].npy"),
    (RESULTS_ROOT / "1/imid[100]_intermediate_imgs[uint8].npy", RESULTS_ROOT / "scratch/imid[100]_intermediate_imgs[uint8].npy"),
    (RESULTS_ROOT / "1/imid[1]_intermediate_imgs[uint8].npy"  , RESULTS_ROOT / "scratch/imid[1]_intermediate_imgs[uint8].npy"  ),
    (RESULTS_ROOT / "27/imid[33]_intermediate_imgs[uint8].npy", RESULTS_ROOT / "scratch/imid[33]_intermediate_imgs[uint8].npy" ),
    (RESULTS_ROOT / "27/imid[70]_intermediate_imgs[uint8].npy", RESULTS_ROOT / "scratch/imid[70]_intermediate_imgs[uint8].npy" ),
]


def _load_frames(idx: int):
    imgid = INPUT_IMAGES[idx].stem[-3:]  # Path 객체는 .stem 사용
    imgid = imgid.lstrip("0")
    npy_path = NPY_PATHS[idx]

    arr = np.load(npy_path[0], mmap_mode="r")
    arr_scratch = np.load(npy_path[1], mmap_mode="r")
    print(f"Loaded: {npy_path[0]} | shape={arr.shape}, dtype={arr.dtype}")
    return arr_scratch, arr


def show_epoch(ep: int, frames_scratch, frames, epochs):
    """슬라이더 변경 시: 해당 epoch 프레임 1장을 반환"""
    idx = max(1, min(epochs, int(ep))) - 1  # 1-based -> 0-based
    frame_scratch = frames_scratch[idx]  # [H,W,C], uint8
    frame = frames[idx]                  # [H,W,C], uint8
    return frame_scratch, frame


# 갤러리 선택 → 상태 및 출력/가시성/슬라이더 갱신
def on_input_select(evt: gr.SelectData, ep: int, epochs):
    idx = int(evt.index)
    frames_scratch, frames = _load_frames(idx)
    epochs = frames.shape[0]
    img_s, img_o = show_epoch(ep, frames_scratch, frames, epochs)

    return (
        gr.update(value=img_s, visible=True),   # img_scratch
        gr.update(value=img_o, visible=True),   # img_ours
        idx, frames_scratch, frames, epochs,    # states
        gr.update(visible=False),               # placeholder_group 숨김
        gr.update(visible=True),                # output_group 표시
        gr.update(maximum=epochs, value=min(ep, epochs), interactive=True)  # slider
    )


with gr.Blocks(title="Enhance INR with Transfer Learning", css="CSS/main.css", fill_height=True) as demo:

    # ───────────────────────────────
    # ① Title Block
    # ───────────────────────────────
    with gr.Row(elem_id="block_title"):
        gr.HTML("""
        <div style="width:100%; text-align:center; padding:40px 0; background-color:#f6f8fa;">
            <h1 style="font-size:4rem; margin-bottom:0.3em;">Enhance INR with Transfer Learning</h1>
            <h2 style="margin-top:0;">Euijun Lee, Minseo Kim, Sung-Ho Bae, Chaoning Zhang</h2>
            <h2 style="margin-top:0;">[IEEE Access 2025-09]</h2>
        </div>
        """)

    # ───────────────────────────────
    # ② INR Explanation Block
    # ───────────────────────────────
    with gr.Row(elem_id="block_inr_desc"):
        gr.Markdown(r"""
## What is Implicit Neural Representation (INR)?

**Implicit Neural Representation (INR)** encodes signals such as images, shapes, or scenes directly as *continuous coordinate-to-value functions* learned by neural networks instead of storing them as discrete pixels.  

An INR model takes a spatial coordinate (x, y) as input 
and outputs the corresponding pixel value (r, g, b) in RGB space.  
This mapping is mathematically formulated as:

$$
\mathbf{c} = M(\mathbf{x}; \theta)
= W_n \, \sigma \big( W_{n-1} \, \sigma( \cdots \sigma( W_1 \mathbf{x} + \mathbf{b}_1 ) \cdots ) + \mathbf{b}_{n-1} \big) + \mathbf{b}_n
$$

**Key advantages**
- *Resolution-free* signal reconstruction  
- Compact, differentiable, and continuous representation  
- Smooth interpolation and fine-grained control for image restoration  
""")

    # ───────────────────────────────
    # ③ Our Contribution Block
    # ───────────────────────────────
    with gr.Row(elem_id="block_contrib"):
        gr.Markdown("""
        ## Our Contribution — Enhancing INR via Transfer Learning

        Our work demonstrates that **transfer learning can directly benefit per-instance INR models** even though each INR is trained on a single image.  
        By pre-training on a structurally complex image and then fine-tuning on a new target, we achieve faster and more balanced learning across all frequency bands.

        **Core insights from the paper (IEEE Access 2025 – 09)**  
        - Transfer learning accelerates INR convergence and reduces spectral bias  
        - High-frequency details are captured up to **29 % faster**, improving edge and texture reconstruction  
        - Source images with **high edge density and low homogeneity** yield better transfer performance  
        - A two-stage procedure (pre-training → fine-tuning) requires only a single source image  
        - Applicable to diverse INR architectures (SIREN, FINER, GAUSS, WIRE variants)

        **Takeaway Message**  
        Even for single-instance image representation tasks, *knowledge transfer from one signal to another* serves as a simple yet effective tool  
        to boost learning efficiency and reconstruction quality :contentReference[oaicite:1]{index=1}.
        """)

    # ───────────────────────────────
    # ④ Interactive Demo Block (기존 코드)
    # ───────────────────────────────
    gr.HTML("""
    <div style="margin-left: 60px;">
        <h2>Interactive Demonstration</h2>
    </div>
    """)
    with gr.Row(elem_id="block_demo"):
        
        with gr.Row():
            # 왼쪽: 입력 갤러리
            in_gallery = gr.Gallery(
                label="Input Gallery (pick any)",
                value=[str(p) for p in INPUT_IMAGES if p.exists()],
                columns=4,
                height=360,
                allow_preview=True,
                elem_id="input_gallery",
            )

            # 오른쪽: 슬라이더 + 이미지
            with gr.Column(elem_id="right_col"):
                epoch_slider = gr.Slider(minimum=1, maximum=2000, value=1, step=1,
                                        label="Epoch", interactive=False)

                with gr.Group(visible=True) as placeholder_group:
                    with gr.Row():
                        tb_scratch = gr.Textbox(label="Scratch", value="Scratch", interactive=False, elem_id="tb_scratch")
                        tb_ours    = gr.Textbox(label="Ours", value="Transfer Learning(Ours)", interactive=False, elem_id="tb_ours")

                with gr.Group(visible=False) as output_group:
                    with gr.Row(elem_id="output_row"):
                        img_scratch = gr.Image(label="Scratch", interactive=False, visible=False, elem_id="output_scratch")
                        img_ours    = gr.Image(label="Ours", interactive=False, visible=False, elem_id="output_ours")

    # 초기 상태 정의 (원하면 유지, 단 화면엔 안 보임)
    init_idx = 0
    init_frames_scratch, init_frames = _load_frames(init_idx)
    init_epochs = init_frames.shape[0]

    state_idx = gr.State(init_idx)
    state_frames = gr.State(init_frames)
    state_frames_scratch = gr.State(init_frames_scratch)
    state_epochs = gr.State(init_epochs)

    # ⛔ 초기 이미지 로드는 하되, 화면엔 텍스트만 보이도록 demo.load는 생략하거나 아래처럼 텍스트만 유지
    # (굳이 demo.load로 이미지 값을 보낼 필요 없음)

    # 슬라이더 변경 → 이미지 갱신 (이미 선택된 뒤에만 유효)
    epoch_slider.change(
        fn=show_epoch,
        inputs=[epoch_slider, state_frames_scratch, state_frames, state_epochs],
        outputs=[img_scratch, img_ours],
    )

    in_gallery.select(
        fn=on_input_select,
        inputs=[epoch_slider, state_epochs],
        outputs=[
            img_scratch, img_ours,                 # 1,2
            state_idx, state_frames_scratch, state_frames, state_epochs,  # 3~6
            placeholder_group, output_group,       # 7,8  ⬅️ 그룹 가시성 토글
            epoch_slider                            # 9
        ],
    )
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)