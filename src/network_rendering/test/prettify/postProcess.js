import * as THREE from 'three';
import * as tf from '@tensorflow/tfjs'

import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

// Selection
const BLOOM_SCENE = 1;
const bloomLayer = new THREE.Layers(); 
bloomLayer.set(BLOOM_SCENE);
const darkMaterial = new THREE.MeshBasicMaterial({color: 'black'}); // Unbloomed material
const materials = {}; // For bookkeeping materials

// Adds discrete Unreal Bloom passes strength proportional to the risk information at an entity
export function addDiscreteBloom(nLevels, riskCov, scene, camera, renderer, bloomParams){

    const renderScene = new RenderPass( scene, camera );    
    const outputPass = new OutputPass();
    let bloomPass, mixPass;   
    
    const bloomComposer = new EffectComposer( renderer );
    bloomComposer.addPass( renderScene );
    bloomComposer.renderToScreen = false;
    const finalComposer = new EffectComposer( renderer );
    finalComposer.addPass( renderScene );

    for (let i = 0; i < nLevels; i++) {
        
        bloomPass = new UnrealBloomPass( new THREE.Vector2( bloomParams.innerWidth, bloomParams.innerHeight ), 1.5, 0.4, 0.85 );
        bloomPass.threshold = bloomParams.threshold;
        bloomPass.strength = bloomParams.strength;
        bloomPass.radius = bloomParams.radius;  
        
        bloomComposer.addPass( bloomPass );

        // Select Objects here
        
    }

    mixPass = new ShaderPass( 
        new THREE.ShaderMaterial( {
            uniforms: {
                baseTexture: { value: null }, // Original Texture
                bloomTexture: { value: bloomComposer.renderTarget2.texture } // Bloom Texture
            },
            vertexShader: document.getElementById( 'vertexShader' ).textContent,
            fragmentShader: document.getElementById( 'fragmentShader' ).textContent
            //defines: {}
        } ), 'baseTexture'
    );
    //mixPass.needsSwap = true;          
    
    finalComposer.addPass( mixPass );
    
    finalComposer.addPass( outputPass );       

    return [bloomComposer, finalComposer]
}

// Set material of obj to dark if it is not part of bloom
export function nonBloomed(obj) {
    if (obj.isMesh && (bloomLayer.test(obj.layers) === false)) {
        //console.log(materials)
        materials[obj.uuid] = obj.material;
        obj.material = darkMaterial;
    }
}

// Restore the original material
export function restoreMaterial(obj) {
    if (materials[obj.uuid]) {
        obj.material = materials[obj.uuid];
        delete  materials[obj.uuid];
    }
}

// Given array of numbers assigns them into discrete groups
export function discretisize(arr, nGroups, {method="linear"} = {}) {

    const groupAssgn = Array(arr.length).fill(0);
    let sortArr = [...arr];
    console.assert(nGroups < arr.length, "Number of discretized groups is larger than array size!")

    if (method == "rank"){
        // Discretize into similar size groups
        sortArr = [...argsort(arr)];
    } 
    // method == "linear"
    // Discretize into similar values
    const minVal = Math.min(...arr);
    const maxVal = Math.max(...arr);
    const width = (maxVal - minVal) / nGroups;
    let group=0, i=0;

    arr.forEach(element => {
        
        group = Math.floor((element - minVal) / width);
        if (group == nGroups) {group -= 1}
        groupAssgn[i] = group;
        i++;
    })
        
    
    return groupAssgn
}

function argsort(arr) {
    return arr.map((value, index) => [value, index])
      .sort((a, b) => a[0] - b[0]) // comparison function for normal sort
      .map(pair => pair[1]);
  }

