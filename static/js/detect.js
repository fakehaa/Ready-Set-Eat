const captureButton = document.getElementById('capture-btn');
const loaderContainer = document.getElementById('loader-container');

captureButton.addEventListener('click', () => {
    loaderContainer.style.display = 'flex';
    setTimeout(() => {
        loaderContainer.style.display = 'none';
    }, 850);
});
