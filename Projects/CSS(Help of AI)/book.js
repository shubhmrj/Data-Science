// Create a new file: static/js/book.js

document.addEventListener('DOMContentLoaded', function() {
    // Get book elements
    const book = document.getElementById('book');
    const frontCover = document.getElementById('front-cover');
    const pages = Array.from(document.querySelectorAll('.page'));
    const bookContainer = document.getElementById('book-container');
    const statusIndicator = document.getElementById('status-indicator');
    const hints = Array.from(document.querySelectorAll('[id^="hint-"]'));
    
    // Control buttons
    const controls = document.createElement('div');
    controls.className = 'controls';
    controls.innerHTML = `
        <button class="control-btn" id="prev-btn">← Previous</button>
        <button class="control-btn" id="next-btn">Next →</button>
        <button class="control-btn" id="toggle-book-btn">Close Book</button>
    `;
    document.querySelector('.scene').appendChild(controls);
    
    const prevBtn = document.getElementById('prev-btn');
    const nextBtn = document.getElementById('next-btn');
    const toggleBookBtn = document.getElementById('toggle-book-btn');
    
    // State variables
    let currentPage = 0;
    let isBookOpen = false;
    let maxPages = pages.length;
    
    // Initialize book state
    function initBook() {
        // Position all pages to be stacked initially
        pages.forEach((page, index) => {
            page.style.transform = 'rotateY(0deg)';
            page.style.zIndex = (14 - index).toString();
        });
        updateControls();
        
        // Show first page hint after a delay
        setTimeout(() => {
            if (hints[0]) hints[0].style.opacity = '1';
        }, 3000);
    }
    
    // Open the book function
    function openBook() {
        frontCover.style.transform = 'rotateY(-180deg)';
        bookContainer.style.transform = 'translateX(50%)';
        document.querySelector('.book-shadow').style.width = '80%';
        isBookOpen = true;
        toggleBookBtn.textContent = 'Close Book';
        updateControls();
        
        // Show status briefly
        showStatus('Book opened');
    }
    
    // Close the book function
    function closeBook() {
        // Return all pages to initial position
        pages.forEach(page => {
            page.style.transform = 'rotateY(0deg)';
        });
        
        // Close the cover after a slight delay to allow pages to settle
        setTimeout(() => {
            frontCover.style.transform = 'rotateY(0deg)';
            bookContainer.style.transform = 'translateX(0)';
            document.querySelector('.book-shadow').style.width = '100%';
        }, 300);
        
        isBookOpen = false;
        currentPage = 0;
        toggleBookBtn.textContent = 'Open Book';
        updateControls();
        
        // Hide all hints
        hints.forEach(hint => hint.style.opacity = '0');
        
        // Show status briefly
        showStatus('Book closed');
    }
    
    // Turn to next page
    function nextPage() {
        if (!isBookOpen) {
            openBook();
            return;
        }
        
        if (currentPage < maxPages) {
            // Hide current hint
            if (hints[currentPage]) hints[currentPage].style.opacity = '0';
            
            pages[currentPage].style.transform = 'rotateY(-180deg)';
            currentPage++;
            
            // Show next hint after turning
            setTimeout(() => {
                if (currentPage < maxPages && hints[currentPage]) {
                    hints[currentPage].style.opacity = '1';
                }
            }, 1000);
            
            updateControls();
            showStatus(`Page ${currentPage} of ${maxPages}`);
        }
    }
    
    // Turn to previous page
    function prevPage() {
        if (!isBookOpen || currentPage === 0) return;
        
        // Hide current hint
        if (hints[currentPage-1]) hints[currentPage-1].style.opacity = '0';
        
        currentPage--;
        pages[currentPage].style.transform = 'rotateY(0deg)';
        
        // Show current hint after turning back
        setTimeout(() => {
            if (hints[currentPage-1]) hints[currentPage-1].style.opacity = '1';
        }, 1000);
        
        updateControls();
        showStatus(`Page ${currentPage} of ${maxPages}`);
    }
    
    // Update control buttons based on current state
    function updateControls() {
        prevBtn.disabled = !isBookOpen || currentPage === 0;
        nextBtn.disabled = currentPage >= maxPages;
        
        prevBtn.style.opacity = prevBtn.disabled ? '0.5' : '1';
        nextBtn.style.opacity = nextBtn.disabled ? '0.5' : '1';
    }
    
    // Show status message briefly
    function showStatus(message) {
        statusIndicator.textContent = message;
        statusIndicator.style.opacity = '1';
        
        setTimeout(() => {
            statusIndicator.style.opacity = '0';
        }, 2000);
    }
    
    // Attach event listeners
    toggleBookBtn.addEventListener('click', () => {
        if (isBookOpen) {
            closeBook();
        } else {
            openBook();
        }
    });
    
    nextBtn.addEventListener('click', nextPage);
    prevBtn.addEventListener('click', prevPage);
    
    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowRight') {
            nextPage();
        } else if (e.key === 'ArrowLeft') {
            prevPage();
        } else if (e.key === 'Escape') {
            closeBook();
        } else if (e.key === 'Enter' || e.key === ' ') {
            if (!isBookOpen) openBook();
        }
    });
    
    // Click on page edges to turn
    pages.forEach((page, index) => {
        const pageEdge = page.querySelector('.page-edge');
        if (pageEdge) {
            pageEdge.addEventListener('click', (e) => {
                e.stopPropagation();
                if (currentPage === index) {
                    nextPage();
                }
            });
        }
        
        // Add hover effect for page turning
        page.addEventListener('mouseenter', () => {
            if (currentPage === index && isBookOpen) {
                page.style.transform = 'rotateY(-15deg)';
            }
        });
        
        page.addEventListener('mouseleave', () => {
            if (currentPage === index && isBookOpen) {
                page.style.transform = 'rotateY(0deg)';
            }
        });
    });
    
    // Click on cover to open book
    frontCover.addEventListener('click', () => {
        if (!isBookOpen) {
            openBook();
        }
    });
    
    // Initialize the book
    initBook();
    
    // Add helpful instructions
    const instructions = document.createElement('div');
    instructions.className = 'instructions';
    instructions.innerHTML = `
        <div style="position: absolute; top: 20px; right: 20px; background: rgba(0,0,0,0.5); 
                    padding: 10px; border-radius: 5px; color: white; font-size: 0.8rem;">
            <p>Click the cover or use arrow keys to navigate</p>
            <p>ESC: Close book | ENTER: Open book</p>
        </div>
    `;
    document.querySelector('.scene').appendChild(instructions);
    
    // Show the instructions for 5 seconds then fade out
    setTimeout(() => {
        instructions.style.opacity = '0';
        setTimeout(() => instructions.remove(), 1000);
    }, 5000);
});