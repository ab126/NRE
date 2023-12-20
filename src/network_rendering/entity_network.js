import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import * as data from './saves/net_data2.json' assert {type: 'json'}; 

let camera, scene, renderer, controls;
const radius = 0.05

init();
animate();

function init(){    

    // Read planar positions
    const {pos, topologyEdges, risk_mean, risk_cov, funcEdges} = data
    console.log(pos)
    
    // Scene & Camera
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.1, 1000 );
    camera.position.z = 4;
    scene.add(camera)

    // Lights
    const light = new THREE.SpotLight( 0xffffff, 4.5 );
    light.position.set( 0, 0, 5 );
    light.castShadow = true;
    scene.add( new THREE.AmbientLight( 0xf0f0f0, 1 ) );
    scene.add( light );

    //Plane
    const planeGeometry = new THREE.PlaneGeometry( 20, 20 );
    const planeMaterial = new THREE.MeshStandardMaterial( { color: 0xd4d4d4 } )
    const plane = new THREE.Mesh( planeGeometry, planeMaterial );
    plane.receiveShadow = true;
    scene.add( plane );

    // Grid Lines
    const helper = new THREE.GridHelper( 5, 10 );
    helper.position.z = 0.01;
    helper.rotateX(Math.PI / 2)
    helper.material.opacity = 0.25;
    helper.material.transparent = true;
    scene.add( helper );

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize( window.innerWidth, window.innerHeight );
    renderer.shadowMap.enabled = true;
    document.body.appendChild( renderer.domElement );

    // Geometries & Material
    
    // Nodes
    const nodeMaterial = new THREE.MeshStandardMaterial( {color: 0x242323} );

    const nodes = [];
    Object.keys(pos).forEach(element => {
        nodes.push(makeNode(element, nodeMaterial, radius))
    });

    // Edges
    const lineMaterial = new THREE.LineBasicMaterial({
        color: 0xffffff
    });

    const edgePoints = makeEdgePoints(topologyEdges)
    const edgeGeometry = new THREE.BufferGeometry().setFromPoints( edgePoints );
    const lines = new THREE.Line( edgeGeometry, lineMaterial );
    scene.add( lines );

    const controls = new OrbitControls( camera, renderer.domElement );

    /**
     * Makes and adds entity as nodes to the scene
     * @param {*} name 
     * @param {*} material
     * @param {*} radius Node raidus
     */
    function makeNode(name, material, radius = 0.05) {

        const geometry_dodec = new THREE.DodecahedronGeometry( radius );
        const dodec = new THREE.Mesh( geometry_dodec, material );
        dodec.position.x = pos[name][0]
        dodec.position.y = pos[name][1]
        dodec.position.z = risk_mean[name]
        
        scene.add( dodec );

        return dodec
    }

    /**
     * Makes edge defining points
     * @param {*} src Source entity
     * @param {*} dst Destination entity
     */
    function makeEdgePoints(edges){
        const edgePoints = []
        var src, dst
        for (let i = 0; i < edges.length; i++) {
            [src, dst] = edges[i]
            edgePoints.push( new THREE.Vector3( pos[src][0], pos[src][1], risk_mean[src] ) )
            edgePoints.push( new THREE.Vector3( pos[dst][0], pos[dst][1], risk_mean[dst] ) )
        }
        return edgePoints
    }
}



function animate() {
    const time = Date.now() * 0.001;
    
    //cube.rotation.y = time * 0.4;
	
	//controls.update();

	renderer.render( scene, camera );

    requestAnimationFrame( animate );
}




