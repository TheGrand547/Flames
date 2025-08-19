# Flames
 
Game in progress with raw(ish) OpenGL, mostly from the ground up following a class in real-time rendering.

Going for a vaguely cartoony aesthetic currently but unsure how it'll develop.

# Compilation
Requires GLEW, FreeGLUT, and GLM to compile, only tested on Windows with Visual Studio 2022

Uses: https://github.com/assimp/assimp for Model/Scene Loading

Uses: https://github.com/ocornut/imgui for UI stuff

Uses: https://github.com/Lek-sys/LeksysINI for ini reading because I couldn't be bothered. Had to *slightly* modify it by the conditions on lines 186 and 196 always true, so that depreciated c++ features(`std::not1` and `std::ptr_fun`) weren't used.

Uses: https://github.com/tuxalin/procedural-tileable-shaders?tab=readme-ov-file for some noise shader stuff
Using OpenGL Core Profile 4.6
