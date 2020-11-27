//
// Created by root on 2020/11/26.
//

#ifndef CUDADEMO_INTERACTIONS_H
#define CUDADEMO_INTERACTIONS_H

#include "kernel.h"
#include "stdio.h"
#include "stdlib.h"
#include "GL/glew.h"
#include "GL/freeglut.h"

#define W 640
#define H 640
#define DT 1.f
#define MAX(x, y) (((x) > (y)) ? (x) : (y))

float *d_temp = 0;
int iterationCount = 0;
BC bc = {
        W / 2, H / 2, W / 10.f, 150, 212.f, 70.f, 0.f
};

void keyboard(unsigned char key, int x, int y) {
    if (key == 'S') {
        bc.t_s += DT;
    }
    if (key == 's') {
        bc.t_s -= DT;
    }
    if (key == 'A') {
        bc.t_a += DT;
    }
    if (key == 'a') {
        bc.t_a -= DT;
    }
    if (key == 'G') {
        bc.t_g += DT;
    }
    if (key == 'g') {
        bc.t_g -= DT;
    }
    if (key == 'R') {
        bc.rad += DT;
    }
    if (key == 'r') {
        bc.rad -= DT;
    }
    if (key == 'C') {
        bc.chamfer++;
    }
    if (key == 'c') {
        bc.chamfer--;
    }
    if (key == 'z') {
        resetTemperature(d_temp, W, H, bc);
    }
    if (key == 27) { // esc
        exit(0);
    }
    glutPostRedisplay();
}

void mouse(int button, int state, int x, int y) {
    bc.x = x;
    bc.y = y;
    glutPostRedisplay();
}

void idle(void ) {
    iterationCount++;
    glutPostRedisplay();
}

void printInstructions() {
    printf("Temperature Visualizer\n");
    printf("Relocate source with mouse click\n");
    printf("Change source temperature (-/+): s/S\n");
    printf("Change air temperature (-/+): a/A\n");
    printf("Change ground temperature (-/+): g/G\n");
    printf("Change pipe radius (-/+): r/R\n");
    printf("Change chamfer (-/+): c/C\n");
    printf("Reset to air temperature: z\n");
    printf("esc: close graphics window\n");
}

#endif //CUDADEMO_INTERACTIONS_H
