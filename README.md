# AXI Visualizer

AXI Visualizer is a complete software 3D graphics pipeline and engine written with Python and OpenCL.

It uses scanline rasterization kernels which can be run on the CPU or the GPU compute units.

See a [full feature list](https://agentxindustries.neocities.org/Visualizer/Features.html) and some [demo clips](http://axi.x10.mx/Visualizer).

Main dependencies are Numpy, Numexpr, PIL, and PyOpenCL.

Currently the engine is used in AXI Animator, a simple and accessible rigging / skinning / posing tool.

The '3xxx.py' files are main scripts which can be directly run.
'AXI_Visualizer.py' can also be directly run.

Here is a sample render showcasing many features of the Visualizer.
![Orchestra of Legends](https://agentxindustries.neocities.org/Backgrounds/Visualizerbg4.png)
For this render, all models were imported as .obj files, then skinned, rigged, and posed with AXI Animator. The trees, grass clumps, and violin strings use alpha test shaders. The terrain texture uses trilinear filtering. Global Illumination is present and visible on the grass below the red panel picking up its color, and the sphere in the far right picking up green from the grass. A double-cascade shadow map was used for visual fidelity. A reflection cubemap is applied on the globe in the left. Full-screen blur highlights bright areas and creates a softer feel. These features are all created from scratch and implemented using OpenCL and Python.
