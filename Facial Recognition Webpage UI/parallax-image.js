const teamImage = document.getElementById('team-image');
const container = document.querySelector('.parallax-container');

window.addEventListener('scroll', () => {
    const containerRect = container.getBoundingClientRect();
    const containerHeight = container.offsetHeight;
    const imageHeight = teamImage.offsetHeight;
    const windowHeight = window.innerHeight;

    // Only move image if container is in viewport
    if (containerRect.top < windowHeight && containerRect.bottom > 0) {
        // Progress: 0 = top of viewport hits container, 1 = bottom of container reaches top of viewport
        let progress = (windowHeight - containerRect.top) / (windowHeight + containerHeight);
        progress = Math.min(Math.max(progress, 0), 1);

        // Move image only halfway
        const maxTranslate = (imageHeight - containerHeight) / 2; // half of the extra image height
        const translateY = progress * maxTranslate;

        teamImage.style.transform = `translateY(-${translateY}px)`;
    }
});
