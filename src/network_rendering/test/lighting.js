import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

let camera, scene, renderer, controls;
let cube, dodec;


init();
animate();

function init(){

    
    // Scene & Camera
    scene = new THREE.Scene();
    scene.background = new THREE.Color( 0xf0f0f0 );
    camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.1, 1000 );
    camera.position.z = 3;
    scene.add( camera );

    // Geometries
    //scene.add( new THREE.AmbientLight( 0xf0f0f0, 3 ) );
    const light = new THREE.SpotLight( 0xffffff, 4.5 );
    light.position.set( 0, 0, 5 );
    light.castShadow = true;
    scene.add( light );

    const planeGeometry = new THREE.PlaneGeometry( 5, 5 );
    const planeMaterial = new THREE.MeshStandardMaterial( { color: 0x00ff00 } )//new THREE.ShadowMaterial( { color: 0xf0f0f0, opacity: 0.2 } );

    const plane = new THREE.Mesh( planeGeometry, planeMaterial );
    plane.receiveShadow = true;
    scene.add( plane );

    const helper = new THREE.GridHelper( 5, 10 );
    helper.position.z = 0.01;
    helper.rotateX(Math.PI / 2)
    helper.material.opacity = 0.25;
    helper.material.transparent = true;
    scene.add( helper );

    // Cube
    const geometry = new THREE.BoxGeometry( 0.5, 0.5, 0.5 ); 
    const material =  new THREE.MeshStandardMaterial( { color: 0xa9b823 } );
    const cube = new THREE.Mesh( geometry, material ); 
    cube.position.z = 1;
    cube.castShadow = true;
    scene.add( cube );

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize( window.innerWidth, window.innerHeight );
    renderer.shadowMap.enabled = true;
    document.body.appendChild( renderer.domElement );
    
    const controls = new OrbitControls( camera, renderer.domElement );
}

function animate() {
    const time = Date.now() * 0.001;
    
    //cube.rotation.y = time * 0.4;
	
	//controls.update();

	renderer.render( scene, camera );

    requestAnimationFrame( animate );
}





