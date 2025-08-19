import * as THREE from 'three';
import * as tf from '@tensorflow/tfjs'

import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

// Selection
const BLOOM_SCENE = 1;
const darkMaterial = new THREE.MeshBasicMaterial({color: 'black'}); // Unbloomed material
const materials = {}; // For bookkeeping materials

// Adds the bloom effect to entities according to their risk variance ranks. 
export function addEntityBloom(riskCov, entityGroup, scene, camera, renderer, bloomParams){

    const entityMaterials = [];
    const renderScene = new RenderPass( scene, camera );      
    const bloomComposer = new EffectComposer( renderer );
    bloomComposer.addPass( renderScene );

    // Rank the nodes
    const nodeVar = Array.from(riskCov, (elem, i) => elem[i]);
    const maxVar = Math.max(...nodeVar);
    const minVar = Math.min(...nodeVar);
    //console.log(maxVar);
    
    // Rank nodes according to variances from low to high
    const sortInd = argsort(nodeVar);
    const nodeBloomRanks = Array(sortInd.length).fill(0);
    sortInd.forEach((elem, i) => {nodeBloomRanks[elem] = i});
    //console.log(riskCov);
    //console.log(nodeBloomRanks);

    const bloomLayer = new THREE.Layers();
    bloomLayer.set(BLOOM_SCENE); 
    bloomLayer.idx = BLOOM_SCENE;
    
    const bloomPass = new UnrealBloomPass( new THREE.Vector2( bloomParams.innerWidth, bloomParams.innerHeight ),
        bloomParams.strength, bloomParams.radius, bloomParams.threshold );      
    bloomComposer.addPass( bloomPass );

    // Specify node emmissive multiplier    
    entityGroup.children.forEach((node, i) => {
        //console.log(node.material);
        node.layers.enable(BLOOM_SCENE);
        
        node.material.emissiveMult = Math.pow( minVar / nodeVar[i], 3);
        node.material.emissiveIntensity = bloomParams.emmisiveIntensity * node.material.emissiveMult;
       
    });
       
    bloomComposer.renderToScreen = false;   

    const mixPass = new ShaderPass( 
        new THREE.ShaderMaterial( {
            uniforms: {
                baseTexture: { value: null }, // Original Texture
                bloomTexture: { value: bloomComposer.renderTarget2.texture } // Bloom Texture
            },
            vertexShader: document.getElementById( 'vertexShader' ).textContent,
            fragmentShader: document.getElementById( 'fragmentShader' ).textContent
        } ), 'baseTexture'
    );     

    const finalComposer = new EffectComposer( renderer );
    finalComposer.addPass( renderScene );
    
    finalComposer.addPass( mixPass );

    const outputPass = new OutputPass();
    finalComposer.addPass( outputPass );       

    return [bloomComposer, finalComposer, bloomPass, bloomLayer]
}


/**
 * Same function for hierarchically clustered entities
 * @param {Array<Array<float>>} riskCov Covariance Matrix of Risk estimates
 * @param {THREE.Group} clusterGroup Group of node groups called 'clusters'
 * @param {Array<string>} clusAssignments Array of cluster indices of all entities
 * @param {Array<int>} entityIndexInClus Array of within cluster indices of all entities
 * @param {Array<string>} namesArr Array of entity names
 * @param {*} scene 
 * @param {*} camera 
 * @param {*} renderer 
 * @param {*} bloomParams 
 * @returns {Array<*>} Returns the bloomComposer, finalComposer, bloomPass, bloomLayer
 */
export function addEntityBloomClustered( riskCov, clusterGroup, clusAssignments, entityIndexInClus, namesArr,
        scene, camera, renderer, bloomParams){
    const entityMaterials = [];
    const renderScene = new RenderPass( scene, camera );      
    const bloomComposer = new EffectComposer( renderer );
    bloomComposer.addPass( renderScene );

    // Rank the nodes
    const nodeVar = Array.from(riskCov, (elem, i) => elem[i]);
    const maxVar = Math.max(...nodeVar);
    const minVar = Math.min(...nodeVar);
    //console.log(maxVar);
    
    // Rank nodes according to variances from low to high
    const sortInd = argsort(nodeVar);
    const nodeBloomRanks = Array(sortInd.length).fill(0);
    sortInd.forEach((elem, i) => {nodeBloomRanks[elem] = i});
    //console.log(riskCov);
    //console.log(nodeBloomRanks);

    const bloomLayer = new THREE.Layers();
    bloomLayer.set(BLOOM_SCENE); 
    bloomLayer.idx = BLOOM_SCENE;
    
    const bloomPass = new UnrealBloomPass( new THREE.Vector2( bloomParams.innerWidth, bloomParams.innerHeight ),
        bloomParams.strength, bloomParams.radius, bloomParams.threshold );      
    bloomComposer.addPass( bloomPass );

    // Specify node emmissive multiplier    
    // console.log(nodeVar.slice(0, 5));
    entityIndexInClus.forEach( (ind, i) => {
        
        let name = namesArr[i];
        let node = clusterGroup.children[ clusAssignments[ name]].children[ind];
        
        node.layers.enable(BLOOM_SCENE);
        
        node.material.emissiveMult = Math.pow( minVar / nodeVar[i], 3);
        node.material.emissiveIntensity = bloomParams.emmisiveIntensity * node.material.emissiveMult;
        //node.material.emissiveIntensity = 0;
        
    });
       
    bloomComposer.renderToScreen = false;   

    const mixPass = new ShaderPass( 
        new THREE.ShaderMaterial( {
            uniforms: {
                baseTexture: { value: null }, // Original Texture
                bloomTexture: { value: bloomComposer.renderTarget2.texture } // Bloom Texture
            },
            vertexShader: document.getElementById( 'vertexShader' ).textContent,
            fragmentShader: document.getElementById( 'fragmentShader' ).textContent
        } ), 'baseTexture'
    );     

    const finalComposer = new EffectComposer( renderer );
    finalComposer.addPass( renderScene );
    
    finalComposer.addPass( mixPass );

    const outputPass = new OutputPass();
    finalComposer.addPass( outputPass );       

    return [bloomComposer, finalComposer, bloomPass, bloomLayer]

}

// Old
// Adds discrete Unreal Bloom passes strength proportional to the risk information at an entity
export function addDiscreteBloom(nLevels, riskCov, entityGroup, scene, camera, renderer, bloomParams){

    let bloomPass, mixPass;   
    const bloomPasses = [];
    const bloomLayers = [];
    const renderScene = new RenderPass( scene, camera );        
    
    const bloomComposer = new EffectComposer( renderer );
    bloomComposer.addPass( renderScene );

    // Rank the nodes
    const nodeVar = Array.from(riskCov, (elem, i) => elem[i]);
    const sortInd = argsort(nodeVar)
    const nodeBloomRanks = Array(sortInd.length).fill(0);
    sortInd.forEach((elem, i) => {nodeBloomRanks[elem] = i});
    //console.log(riskCov);


    const nNodesPerLevel = Math.ceil(entityGroup.children.length / nLevels);;
    for (let i = 0, inds, strMult, node, layerIdx; i < nLevels; i++) {
        
        let bloomLayer = new THREE.Layers();
        layerIdx = BLOOM_SCENE + i;
        bloomLayer.set(layerIdx);
        
        //console.log(bloomLayer);
        bloomLayers.push(bloomLayer);
        strMult = Math.pow((i+1) / nLevels, 5);
        //console.log(i, strMult);
        if (i === 0) {
            //continue; 
        }
        
        bloomPass = new UnrealBloomPass( new THREE.Vector2( bloomParams.innerWidth, bloomParams.innerHeight ),
         bloomParams.strength * strMult, bloomParams.radius, bloomParams.threshold );      
        bloomComposer.addPass( bloomPass );
        bloomPasses.push(bloomPass);

        // Select Objects 
        inds = findAllIndexesInRange(nodeBloomRanks, i*nNodesPerLevel, (i+1)*nNodesPerLevel);
        //console.log(i, inds)

        // Add nodes at inds to bloomLayer
        inds.forEach(i => {
            node = entityGroup.children[i];
            //console.log('before', node.layers);
            node.layers.enable(layerIdx);
            //console.log('after', node.layers);
        });
        //console.log('Layer after', bloomLayer);

    }
    bloomComposer.renderToScreen = false;   

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

    const finalComposer = new EffectComposer( renderer );
    finalComposer.addPass( renderScene );
    
    finalComposer.addPass( mixPass );

    const outputPass = new OutputPass();
    finalComposer.addPass( outputPass );       

    return [bloomComposer, finalComposer, bloomPasses, bloomLayers]
}

// Set material of obj to dark if it is not part of any bloom layer
export function nonBloomedMulti(obj, bloomLayers) {
    
    const bloomLayerMemberships = Array.from(bloomLayers, bloomLayer => (bloomLayer.test(obj.layers) ) );
    if (obj.isObject3D && (bloomLayerMemberships.indexOf(true) === -1)) { // if its not member of any bloom layer
        //console.log(materials)
        materials[obj.uuid] = obj.material;
        obj.material = darkMaterial;
    }
}

export function nonBloomed(obj, bloomLayer) {
    
    if (obj.isObject3D && (bloomLayer.test(obj.layers) === false)) { // if its not member of bloom layer
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

// Return all the indices of array in range [lower, upper)
function findAllIndexesInRange(array, lower, upper) {
    //console.log(array);
    return array
        .map((elem, index) => ({elem, index}))
        .filter(({elem}) => elem >= lower && elem < upper)
        .map(({index}) => index);
}

