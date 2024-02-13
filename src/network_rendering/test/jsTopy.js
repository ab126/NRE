import $ from 'jquery';

//string
const data_to_pass_in = 'Send this to python';
console.log('Data Sent to python script: ', data_to_pass_in);

$.ajax({
    url: "py2js.py",
    data: data_to_pass_in
});


/*
const xhttp = new XMLHttpRequest();


//const python_process = spawner('python', ['../python/py2js.py', data_to_pass_in]);
xhttp.open("GET", "/py2js.py", false);
xhttp.send();
console.log(xhttp)
console.log(xhttp.response)
*/
