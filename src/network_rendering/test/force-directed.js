import * as tf from '@tensorflow/tfjs'



/**
 * Single Step Force Directed Method for entity layout
 * @param {*} funcAdj Array of functional connectivity graph adjacency matrix
 * @param {*} nodePos Node positions
 * @param {*} t Temperature or the expected step size of displacement
 * @param {*} d_xy Diameter of the node distribution (max - min)
 * @param {*} minDist Minimum distance between entities
 * @param {*} alpha Coefficient of risk mean field
 * @param {*} beta Coefficient controlling graph diversion over spring force
 */
export function singleStepForceDirected(funcAdj, nodePos = null, t = null, d_xy = 1.3, minDist = 0.1, alpha=0.2, beta=1){
    
    const nnodes = funcAdj.length;
    const k = tf.sqrt(1/ nnodes);
    if (nodePos == null){
        nodePos = Array.from({length: nnodes}, () => Math.random() * d_xy - d_xy / 2 );
    }
    if (t == null){
        t = d_xy * 0.1; // Temperatrue or max step size 
    }

    let nodePosTensor = tf.tensor(nodePos);    
    const weightTensor = tf.tensor(funcAdj);
    const delta = tf.expandDims(nodePosTensor, 1).sub(tf.expandDims(nodePosTensor, 0)); 
    //delta.print();
    let distance = tf.norm(delta, undefined, 2);

    // Enforcing min distance 
    distance = distance.clipByValue(minDist, distance.max().arraySync());

    //Forces
    const normDistance = tf.div(distance, k);
    const springForce = tf.dot( weightTensor, normDistance);

    let displacement = tf.einsum("ijk,ij->ik", delta, tf.pow(normDistance, -2).sub(springForce)); // As of now its force times distance
    //displacement = displacement.add() // Risk field

    //Update
    let length = tf.norm(displacement, undefined, 1);
    length = length.clipByValue(0.01, length.max().arraySync());
    const deltaPos = tf.einsum("ij,i->ij", displacement, tf.tensor([t]).div(length) ); // Scale the displacement
    nodePosTensor = nodePosTensor.add(deltaPos);

    //console.log(tf.memory());
    const newNodePos = nodePosTensor.arraySync();

    return newNodePos;
}

