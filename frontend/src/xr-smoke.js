import * as THREE from 'three';
import { VRButton } from './VRButton.js';

const container = document.getElementById('app');

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xe17a1a);

const camera = new THREE.PerspectiveCamera(70, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.set(0, 1.6, 3);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.xr.enabled = true;
container.appendChild(renderer.domElement);
document.body.appendChild(VRButton.createButton(renderer));

const hemi = new THREE.HemisphereLight(0xffffff, 0x3d2a18, 2.2);
scene.add(hemi);

const dir = new THREE.DirectionalLight(0xffffff, 1.1);
dir.position.set(3, 6, 4);
scene.add(dir);

const floor = new THREE.GridHelper(14, 28, 0xffffff, 0x7b2f00);
floor.position.y = 0;
scene.add(floor);

const cube = new THREE.Mesh(
  new THREE.BoxGeometry(0.55, 0.55, 0.55),
  new THREE.MeshStandardMaterial({ color: 0x2fd45f, roughness: 0.35, metalness: 0.05 })
);
cube.position.set(0, 1.55, -2.0);
scene.add(cube);

const leftBall = new THREE.Mesh(
  new THREE.SphereGeometry(0.18, 24, 24),
  new THREE.MeshStandardMaterial({ color: 0x1677ff, emissive: 0x0b2857, emissiveIntensity: 0.5 })
);
leftBall.position.set(-0.8, 1.35, -1.4);
scene.add(leftBall);

const rightBall = new THREE.Mesh(
  new THREE.SphereGeometry(0.18, 24, 24),
  new THREE.MeshStandardMaterial({ color: 0xff2f6d, emissive: 0x5d1127, emissiveIntensity: 0.5 })
);
rightBall.position.set(0.8, 1.35, -1.4);
scene.add(rightBall);

const axes = new THREE.AxesHelper(1.2);
axes.position.set(0, 0.01, -1.2);
scene.add(axes);

renderer.xr.addEventListener('sessionstart', () => {
  console.log('[xr-smoke] session started');
});

renderer.xr.addEventListener('sessionend', () => {
  console.log('[xr-smoke] session ended');
});

function render(time) {
  const t = time * 0.001;
  cube.rotation.x = t * 0.7;
  cube.rotation.y = t * 1.1;
  leftBall.position.y = 1.35 + Math.sin(t * 1.5) * 0.08;
  rightBall.position.y = 1.35 + Math.cos(t * 1.5) * 0.08;
  renderer.render(scene, camera);
}

renderer.setAnimationLoop(render);

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});
