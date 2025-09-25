# <h1 align = "center">TxT2IR: Text-to-Infrared image generation via thermal mask constraints in cross-modal learning</h1>

<p align = "center">Fuchao Wang<sup>a</sup>, Jian Fang<sup>b</sup>, Pengfei Liu<sup>b</sup>, Ronghua Zhang<sup>a</sup>, Yuhuai Peng<sup>a,c,*</sup>, Huaici Zhao<sup>a,b,*</sup>

<p align = "center">a.School of Computer Science and Engineering,  Northeastern University, Shenyang, Liaoning, China</p>
<p align = "center">b.Shenyang Institute of Automation, Chinese Academy of Sciences, Shenyang, Liaoning, China</p>
<p align = "center">c.Strategic Research Department, Zhiyuan Research Institute, Hangzhou, Zhejiang, China</p>

**Abstract** To address the poor quality of infrared images generated from visible images under low-light conditions, we propose TxT2IR, a latent diffusion model (LDM) framework guided by cross-modal textual information. Specifically, (1) we develop an end-to-end text-driven diffusion architecture, TxT2IR, tailored for infrared image generation; (2) we construct TxT2IR-dataset, the first open-source text-infrared paired dataset for low-light scenarios; (3) we introduce a physics-aware thermal mask mechanism and integrate it into the loss function to enhance the thermal radiation consistency. Extensive experiments demonstrate that TxT2IR not only achieves a state-of-the-art FID score of 71.79, but also improves the quality and thermal radiation consistency of infrared images generated under low-light conditions. Furthermore, TxT2IR-generated infrared images exhibit competitive performance in downstream object detection tasks. This study provides a novel technical avenue for research in the field of infrared image generation.

<h2>Architecture</h2>

<img src="figs/TxToIROverview.png" alt="Alt text" title="Architecture" style="zoom: 80%;" />

<h2>Results</h2>

<img src="figs/tab1.png" alt="Alt text" title="DifferentModelMetric" style="zoom: 60%;" />
<img src="figs/genIRImgsv3.png" alt="Alt text" title="DifferentModelMetric" style="zoom: 60%;" />

[dataset](https://drive.google.com/file/d/1uJ-mBWg5o8UucdjavE0GSS26BcA_AmM3/view?usp=drive_link)
(contact us: 2390229@stu.neu.edu.cn)
