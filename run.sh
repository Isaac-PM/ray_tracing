nvcc -O3 -o ray_tracing src/main.cu src/Vec3.cu src/RGBPixel.cu src/Viewport.cu src/PPMImage.cu src/Sphere.cu src/Material.cu src/Hittable.cu src/Renderer.cu && ./ray_tracing

# sudo gprof ray_tracing gmon.out | ./gprof2dot.py -w | dot -Tpng -o output.png
# nvcc -O0 -pg -o ray_tracing src/main.cu src/Vec3.cu src/RGBPixel.cu src/Viewport.cu src/PPMImage.cu src/Sphere.cu src/Material.cu src/Hittable.cu src/Renderer.cu && ./ray_tracing