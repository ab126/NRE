import * as THREE from 'three';




function makeOutline( scene ) {
    const pieOutline = new THREE.Group();
    scene.add(pieOutline);

    const outlineMaterial = new THREE.LineBasicMaterial( { color: 0xffffff } );

    const outerCircle = new THREE.EllipseCurve(
        0,  0,            // ax, aY
        extras.radius + extras.diam_xy/2, extras.radius + extras.diam_xy/2,  // xRadius, yRadius
        0,  2 * Math.PI,  // aStartAngle, aEndAngle
        false,            // aClockwise
        0                 // aRotation
    );
    let points = outerCircle.getPoints( 1000 );
    let pieGeometry = new THREE.BufferGeometry().setFromPoints( points );
    const outerLine = new THREE.Line( pieGeometry, outlineMaterial );
    pieOutline.add(outerLine);

    const innerCircle = new THREE.EllipseCurve(
        0,  0,            // ax, aY
        extras.radius - extras.diam_xy/2, extras.radius - extras.diam_xy/2,  // xRadius, yRadius
        0,  2 * Math.PI,  // aStartAngle, aEndAngle
        false,            // aClockwise
        0                 // aRotation
    );
    points = innerCircle.getPoints( 1000 );
    pieGeometry = new THREE.BufferGeometry().setFromPoints( points );
    const innerLine = new THREE.Line( pieGeometry, outlineMaterial );
    pieOutline.add(innerLine);

    // The Line Segments

    const n_segments = extras.n_cluster - 1;

    for (let i = 0; i < n_segments ; i++) {
        
        const phi = 2 * Math.PI / n_segments;
        let nu_i = phi * i + phi / 2;

        let points = [];
        points.push( new THREE.Vector3( (extras.radius - extras.diam_xy/2) * Math.sin( nu_i),
                                        (extras.radius - extras.diam_xy/2) * Math.cos( nu_i), 0 ) );
        points.push( new THREE.Vector3( (extras.radius + extras.diam_xy/2) * Math.sin( nu_i),
                                        (extras.radius + extras.diam_xy/2) * Math.cos( nu_i), 0 ) );

        const geometry = new THREE.BufferGeometry().setFromPoints( points );
        const line = new THREE.Line( geometry, outlineMaterial );
        pieOutline.add(line);
    }
    return pieOutline
}