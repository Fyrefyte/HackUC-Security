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

// Carousel JS
document.addEventListener('DOMContentLoaded', () => {
    const carouselInner = document.querySelector('.carousel-inner');
    const members = document.querySelectorAll('.member');
    const leftArrow = document.querySelector('.left-arrow');
    const rightArrow = document.querySelector('.right-arrow');

    let currentIndex = 0;

    function updateCarousel() {
        const offset = -currentIndex * 100;
        carouselInner.style.transform = `translateX(${offset}%)`;
        carouselInner.style.transition = 'transform 0.5s ease-in-out';
    }

    leftArrow.addEventListener('click', () => {
        currentIndex = (currentIndex - 1 + members.length) % members.length;
        updateCarousel();
    });

    rightArrow.addEventListener('click', () => {
        currentIndex = (currentIndex + 1) % members.length;
        updateCarousel();
    });

    updateCarousel();
});