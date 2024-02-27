import * as tf from '@tensorflow/tfjs'


/**
 * Single Step Force Directed Method for entity layout
 * @param {*} funcAdj Array of functional connectivity graph adjacency matrix
 * @param {*} nodePos Node positions
 * @param {*} t Temperature or the expected step size of displacement
 * @param {*} diamXY Diameter of the node distribution (max - min)
 * @param {*} minDist Minimum distance between entities
 * @param {*} alpha Coefficient of risk mean field
 * @param {*} beta Coefficient controlling graph diversion over spring force
 */
export function singleStepForceDirected(funcAdj, nodePos = null, t = null, diamXY = 1.3, minDist = 0.001, alpha=0.2, beta=1){
    
    const nnodes = funcAdj.length;
    const k = tf.sqrt(1/ nnodes).mul(diamXY).arraySync();
    if (nodePos == null){
        nodePos = Array.from({length: nnodes}, () => Math.random() * diamXY - diamXY / 2 );
    }
    if (t == null){
        t = diamXY * 0.1; // Temperatrue or max step size 
    }

    let nodePosTensor = tf.tensor(nodePos);    
    const weightTensor = tf.tensor(funcAdj);
    const delta = tf.expandDims(nodePosTensor, 1).sub(tf.expandDims(nodePosTensor, 0)); 
    //delta.print();
    let distance = tf.norm(delta, undefined, 2);

    // Enforcing min distance 
    distance = distance.clipByValue(minDist, distance.max().arraySync());

    //delta.print();
    //Forces
    const normDistance = distance.div(k);
    const springForce = tf.dot( weightTensor, normDistance);
    const normDelta = delta.div( tf.norm(delta, undefined, 2).expandDims(2).clipByValue(minDist, delta.max().arraySync()) );
    //normDelta.print()

    let displacement = tf.einsum("ijk,ij->ik", normDelta, tf.pow(normDistance, -2).sub(springForce.div(200))); // As of now its force times distance
    //let displacement = tf.einsum("ijk,ij->ik", normDelta, tf.pow(normDistance, -2).sub(springForce.div(600)).add(tf.pow(normDistance, -4).div(50) ) ); // As of now its force times distance
    //displacement = displacement.add() // Risk field
    

    //Update
    let length = tf.norm(displacement, undefined, 1);
    length = length.clipByValue(minDist, length.max().arraySync());
    //const deltaPos = tf.einsum("ij,i->ij", displacement, tf.tensor([t]).div(length) ); // Scale the displacement
    const deltaPos = displacement.mul(t / length.max().arraySync());
    nodePosTensor = nodePosTensor.add(deltaPos);

    //console.log(tf.memory());
    const newNodePos = nodePosTensor.arraySync();

    return newNodePos;
}

// Scale the nodePositions to fit the max and min coords (bounds)
export function scaleToBounds(nodePos, bounds){
    let nodePosTensor = tf.tensor(nodePos);
    let newNodePosTensor = tf.tensor(nodePos);

    const minX = tf.min(nodePosTensor, 0);
    const maxX = tf.max(nodePosTensor, 0);
    const upperBounds = tf.tensor(bounds.upper);
    const lowerBounds = tf.tensor(bounds.lower);

    newNodePosTensor = nodePosTensor.sub(minX.expandDims(0)).mul( upperBounds.sub(lowerBounds).expandDims(0) ).div(
        maxX.sub(minX).expandDims(0) ).add(lowerBounds.expandDims(0));

    //console.log(nodePosTensor.arraySync());
    //nodePosTensor.mul(upperBounds.expandDims(0)).print();
    
    return newNodePosTensor.arraySync();
}
