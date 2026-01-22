/**
 * PhotoGeni Collections Page JavaScript
 * Handles all collections functionality including file upload, organization, and navigation
 */

// Global variables
let currentPath = '';
let selectedUsers = [];
let currentItem = null;
let selectedFiles = []; // Store selected files for upload

// Make functions available globally for inline HTML access
window.showUploadModal = function() {
    console.log("Global showUploadModal called");
    // Direct call to internal function instead of toggleUploadZone 
    // which might cause infinite recursion
    $('#upload-zone').addClass('show');
};

// Global function for organizing by faces (needed for backward compatibility)
window.organizeByFaces = function() {
    console.log("Global organizeByFaces called");
    // Call the internal function directly
    organizeByFacesInternal();
};

// Global function for navigation (needed for backward compatibility)
window.navigateToRoot = function() {
    console.log("Global navigateToRoot called");
    loadCollection('');
    return false;
};

// Global function for showing new folder inline (needed for backward compatibility)
window.showInlineNewFolder = function() {
    console.log("Global showInlineNewFolder called");
    // Call the internal function directly without recursive call
    showInlineNewFolderInternal();
};

// DOM Ready
$(document).ready(function() {
    console.log("Collections.js loaded - Document ready");
    
    // Initial load
    loadCollection('');
    
    // Set view preference from localStorage
    const savedView = localStorage.getItem('collectionView') || 'grid';
    if (savedView === 'list') {
        $('#listViewBtn-main').click();
    } else {
        $('#gridViewBtn-main').click();
    }
    
    // Setup event listeners
    setupEventListeners();
});

// Setup all event listeners
function setupEventListeners() {
    console.log("Setting up event listeners");
    
    // Debug logs for critical elements
    console.log("Upload button exists:", $('#uploadBtn-main').length);
    console.log("Side panel upload exists:", $('#side-panel-upload').length);
    console.log("Browse files button exists:", $('#browse-files-btn').length);
    
    // File input change
    $('#file-input').on('change', function() {
        console.log("File input changed, files selected:", this.files.length);
        handleFileSelection(this.files);
    });
    
    // Upload zone drag and drop
    setupDragAndDrop();
    
    // View switching
    $('#gridViewBtn-main').on('click', function() {
        console.log("Grid view button clicked");
        $('#collections-container').removeClass('list-view').addClass('grid-view');
        $('#gridViewBtn-main').addClass('active');
        $('#listViewBtn-main').removeClass('active');
        localStorage.setItem('collectionView', 'grid');
    });
    
    $('#listViewBtn-main').on('click', function() {
        console.log("List view button clicked");
        $('#collections-container').removeClass('grid-view').addClass('list-view');
        $('#listViewBtn-main').addClass('active');
        $('#gridViewBtn-main').removeClass('active');
        localStorage.setItem('collectionView', 'list');
    });
    
    // ALL Upload buttons
    $('#uploadBtn-main, #uploadBtn-header').on('click', function(e) {
        e.preventDefault();
        console.log("Upload button clicked");
        toggleUploadZone();
    });
    
    // Side panel upload area
    $('#side-panel-upload').on('click', function(e) {
        e.preventDefault();
        console.log("Side panel upload clicked");
        toggleUploadZone();
    });
    
    // New folder buttons
    $('#newFolderBtn-main, #newFolderBtn-header').on('click', function(e) {
        e.preventDefault();
        console.log("New folder button clicked");
        showInlineNewFolderInternal();
    });
    
    // Organize button
    $('#organizeBtn').on('click', function(e) {
        e.preventDefault();
        console.log("Organize by faces button clicked");
        organizeByFacesInternal();
    });
    
    // New sidebar organize faces button
    $('#organize-faces-sidebar-btn').on('click', function(e) {
        e.preventDefault();
        console.log("Sidebar organize faces button clicked");
        organizeByFacesInternal();
    });
    
    // View all people button
    $('#view-all-people-btn').on('click', function(e) {
        e.preventDefault();
        console.log("View all people button clicked");
        window.location.href = '/person';
    });
    
    // Breadcrumb home link
    $('#breadcrumb-home').on('click', function(e) {
        e.preventDefault();
        console.log("Breadcrumb home clicked");
        loadCollection('');
    });
    
    // Upload zone buttons
    $('#browse-files-btn').on('click', function(e) {
        e.preventDefault();
        console.log("Browse files button clicked");
        $('#file-input').click();
    });
    
    $('#cancel-upload-btn').on('click', function(e) {
        e.preventDefault();
        console.log("Cancel upload button clicked");
        toggleUploadZone();
    });
    
    $('#upload-files-btn').on('click', function(e) {
        e.preventDefault();
        console.log("Upload files button clicked");
        uploadFiles();
    });
}

// Setup drag and drop functionality
function setupDragAndDrop() {
    console.log("Setting up drag and drop");
    
    const dropZones = [
        document.getElementById('upload-zone'),
        document.getElementById('side-panel-upload')
    ];
    
    dropZones.forEach(zone => {
        if (!zone) {
            console.error("Drop zone element not found");
            return;
        }
        
        zone.addEventListener('dragover', function(e) {
            e.preventDefault();
            e.stopPropagation();
            $(this).addClass('drag-over');
        });
        
        zone.addEventListener('dragleave', function(e) {
            e.preventDefault();
            e.stopPropagation();
            $(this).removeClass('drag-over');
        });
        
        zone.addEventListener('drop', function(e) {
            console.log("Files dropped");
            e.preventDefault();
            e.stopPropagation();
            $(this).removeClass('drag-over');
            
            // Open upload zone if dropped on side panel
            if (this.id === 'side-panel-upload' && !$('#upload-zone').hasClass('show')) {
                toggleUploadZone();
            }
            
            if (e.dataTransfer.files.length > 0) {
                handleFileSelection(e.dataTransfer.files);
            }
        });
    });
}

// Internal function to organize by faces
function organizeByFacesInternal() {
    console.log("Organizing collection by faces...");
    
    // Show loading indicator
    showLoading('Organizing by Faces...');
    
    // Call API to organize by faces
    $.ajax({
        url: '/api/organize-by-faces',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({ collection_path: currentPath }),
        success: function(response) {
            console.log("Organize by faces response:", response);
            hideLoading();
            
            if (response.success) {
                showToast(response.message, 'success');
                // Refresh the collection after organizing
                loadCollection(currentPath);
            } else {
                showToast(response.message || 'Failed to organize by faces', 'error');
            }
        },
        error: function(xhr, status, error) {
            console.error("Error organizing by faces:", error);
            hideLoading();
            showToast('Error organizing by faces', 'error');
        }
    });
}

// Toggle upload zone visibility
function toggleUploadZone() {
    console.log("Toggling upload zone");
    
    const $uploadZone = $('#upload-zone');
    
    if ($uploadZone.hasClass('show')) {
        // Hide the upload zone
        $uploadZone.removeClass('show');
        // Clear any selected files
        $('#file-input').val('');
        $('#preview-container').empty();
        selectedFiles = [];
    } else {
        // Show the upload zone
        $uploadZone.addClass('show');
    }
}

// Handle file selection
function handleFileSelection(fileList) {
    console.log("Handling file selection, count:", fileList.length);
    
    // Convert FileList to array and store
    selectedFiles = Array.from(fileList);
    
    // Clear previous previews
    $('#preview-container').empty();
    
    // Generate previews for each file
    selectedFiles.forEach((file, index) => {
        generateFilePreview(file, index);
    });
}

// Generate preview for a file
function generateFilePreview(file, index) {
    console.log("Generating preview for file:", file.name);
    
    const reader = new FileReader();
    const isImage = file.type.startsWith('image/');
    
    // Create preview element
    const previewEl = $(`
        <div class="file-preview" data-index="${index}">
            <div class="file-preview-content">
                ${isImage ? '<div class="file-preview-image"></div>' : 
                    `<div class="file-preview-icon">
                        <i class="fas ${getFileIconFromMime(file.type)}"></i>
                    </div>`
                }
            </div>
            <div class="file-preview-info">
                <div class="file-preview-name">${file.name}</div>
                <div class="file-preview-size">${formatFileSize(file.size)}</div>
            </div>
            <button class="file-preview-remove" data-index="${index}">
                <i class="fas fa-times"></i>
            </button>
        </div>
    `);
    
    // Add to preview container
    $('#preview-container').append(previewEl);
    
    // Generate image preview if it's an image file
    if (isImage) {
        reader.onload = function(e) {
            previewEl.find('.file-preview-image').css('background-image', `url(${e.target.result})`);
        };
        reader.readAsDataURL(file);
    }
    
    // Set up remove button
    previewEl.find('.file-preview-remove').on('click', function() {
        const index = $(this).data('index');
        removeFilePreview(index);
    });
}

// Remove file preview
function removeFilePreview(index) {
    console.log("Removing file preview at index:", index);
    
    // Remove from DOM
    $(`.file-preview[data-index="${index}"]`).remove();
    
    // Remove from selected files
    selectedFiles = selectedFiles.filter((_, i) => i !== index);
    
    // If no files left, clear input
    if (selectedFiles.length === 0) {
        $('#file-input').val('');
    }
}

// Get file icon based on MIME type
function getFileIconFromMime(mimeType) {
    if (mimeType.startsWith('image/')) return 'fa-image';
    if (mimeType.startsWith('video/')) return 'fa-film';
    if (mimeType.startsWith('audio/')) return 'fa-music';
    if (mimeType.startsWith('text/')) return 'fa-file-alt';
    if (mimeType === 'application/pdf') return 'fa-file-pdf';
    if (mimeType.includes('word')) return 'fa-file-word';
    if (mimeType.includes('excel') || mimeType.includes('spreadsheet')) return 'fa-file-excel';
    if (mimeType.includes('powerpoint') || mimeType.includes('presentation')) return 'fa-file-powerpoint';
    if (mimeType.includes('zip') || mimeType.includes('compressed')) return 'fa-file-archive';
    
    return 'fa-file';
}

// Upload selected files
function uploadFiles() {
    console.log("Uploading files, count:", selectedFiles.length);
    
    if (selectedFiles.length === 0) {
        showToast('No files selected for upload', 'warning');
        return;
    }
    
    // Create form data
    const formData = new FormData();
    
    // Add path
    formData.append('path', currentPath);
    
    // Add files
    selectedFiles.forEach(file => {
        formData.append('files[]', file);
    });
    
    // Show loading
    showLoading('Uploading Files...');
    
    // Upload request
    $.ajax({
        url: '/api/collections/upload',
        type: 'POST',
        data: formData,
        processData: false,
        contentType: false,
        xhr: function() {
            // Custom XHR for progress tracking
            const xhr = new window.XMLHttpRequest();
            xhr.upload.addEventListener('progress', function(e) {
                if (e.lengthComputable) {
                    const percent = Math.round((e.loaded / e.total) * 100);
                    updateLoadingProgress(percent);
                }
            }, false);
            return xhr;
        },
        success: function(response) {
            console.log("Upload response:", response);
            hideLoading();
            
            if (response.success) {
                showToast('Files uploaded successfully', 'success');
                // Close upload zone
                toggleUploadZone();
                // Refresh the collection
                loadCollection(currentPath);
            } else {
                showToast(response.message || 'Failed to upload files', 'error');
            }
        },
        error: function(xhr, status, error) {
            console.error("Error uploading files:", error);
            hideLoading();
            showToast('Error uploading files', 'error');
        }
    });
}

// Show loading indicator
function showLoading(message) {
    console.log("Showing loading:", message);
    
    // Create loading overlay if it doesn't exist
    if ($('#loading-overlay').length === 0) {
        $('body').append(`
            <div id="loading-overlay">
                <div class="loading-content">
                    <div class="spinner"></div>
                    <div class="loading-message">${message}</div>
                    <div class="loading-progress">
                        <div class="progress-bar"></div>
                    </div>
                </div>
            </div>
        `);
    } else {
        // Update message
        $('#loading-overlay .loading-message').text(message);
        // Reset progress
        $('#loading-overlay .progress-bar').css('width', '0%');
    }
    
    // Show loading
    $('#loading-overlay').addClass('show');
}

// Update loading progress
function updateLoadingProgress(percent) {
    $('#loading-overlay .progress-bar').css('width', `${percent}%`);
}

// Hide loading indicator
function hideLoading() {
    $('#loading-overlay').removeClass('show');
}

// Show the inline new folder input - internal implementation
function showInlineNewFolderInternal() {
    console.log("Show inline new folder called");
    
    // Remove any existing inline forms
    $('.inline-form').remove();
    
    // Create the inline form
    const inlineForm = $(`
        <div class="inline-form new-folder-form">
            <input type="text" class="form-control" id="new-folder-name" placeholder="Enter folder name">
            <div class="inline-form-actions">
                <button class="btn btn-light" id="cancel-new-folder-btn">Cancel</button>
                <button class="btn btn-primary" id="create-folder-btn">Create</button>
            </div>
        </div>
    `);
    
    // Add to container
    $('#collections-container').prepend(inlineForm);
    
    // Focus the input
    $('#new-folder-name').focus();
    
    // Set up the buttons
    $('#cancel-new-folder-btn').on('click', function() {
        $('.new-folder-form').remove();
    });
    
    $('#create-folder-btn').on('click', function() {
        createNewFolder();
    });
    
    // Handle enter key
    $('#new-folder-name').on('keypress', function(e) {
        if (e.which === 13) {
            createNewFolder();
        }
    });
}

// Helper function to format file size in human-readable format
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Helper function to show toast notifications
function showToast(message, type = 'info') {
    console.log(`Toast notification: ${message} (${type})`);
    
    // Remove existing toasts
    $('.toast').remove();
    
    // Create toast HTML
    const toast = $(`
        <div class="toast toast-${type}">
            <div class="toast-content">
                <i class="fas ${getToastIcon(type)}"></i>
                <span>${message}</span>
            </div>
            <button class="toast-close">&times;</button>
        </div>
    `);
    
    // Add to body
    $('body').append(toast);
    
    // Show with animation
    setTimeout(() => {
        toast.addClass('show');
    }, 10);
    
    // Close button
    toast.find('.toast-close').on('click', function() {
        toast.removeClass('show');
        setTimeout(() => {
            toast.remove();
        }, 300);
    });
    
    // Auto-close after 5 seconds
    setTimeout(() => {
        toast.removeClass('show');
        setTimeout(() => {
            toast.remove();
        }, 300);
    }, 5000);
}

// Helper function to get toast icon based on type
function getToastIcon(type) {
    switch (type) {
        case 'success': return 'fa-check-circle';
        case 'error': return 'fa-exclamation-circle';
        case 'warning': return 'fa-exclamation-triangle';
        case 'info':
        default: return 'fa-info-circle';
    }
}

// Load collection contents
function loadCollection(path) {
    console.log("Loading collection:", path);
    currentPath = path;
    
    $.ajax({
        url: '/api/collections/list',
        type: 'GET',
        data: { path: path },
        success: function(data) {
            renderCollection(data);
            updateBreadcrumb(path);
        },
        error: function(xhr, status, error) {
            console.error("Error loading collection:", error);
            showToast('Error loading collection', 'error');
        }
    });
}

// Render collection contents
function renderCollection(data) {
    console.log("Rendering collection with", data.items?.length, "items");
    
    const $container = $('#collections-container');
    $container.empty();
    
    if (!data.items || data.items.length === 0) {
        $container.html(`
            <div class="empty-state">
                <i class="fas fa-folder-open"></i>
                <h3>This folder is empty</h3>
                <p>Upload files or create a new folder to get started.</p>
            </div>
        `);
        return;
    }
    
    // Sort items: folders first, then files alphabetically
    const items = [...data.items].sort((a, b) => {
        if (a.type === 'directory' && b.type !== 'directory') return -1;
        if (a.type !== 'directory' && b.type === 'directory') return 1;
        return a.name.localeCompare(b.name);
    });
    
    // Create items
    items.forEach(item => {
        const isFolder = item.type === 'directory';
        const $item = $(`
            <div class="collection-item" data-type="${item.type}" data-path="${item.path}">
                <div class="item-icon">
                    <i class="fas ${isFolder ? 'fa-folder' : getFileIconClass(item.name)}"></i>
                </div>
                <div class="item-details">
                    <div class="item-name">${item.name}</div>
                    <div class="item-meta">
                        ${isFolder ? 'Folder' : item.name.split('.').pop().toUpperCase()}
                        ${!isFolder && item.size ? ' • ' + formatFileSize(item.size) : ''}
                    </div>
                </div>
            </div>
        `);
        
        // Handle item click
        $item.on('click', function() {
            if (isFolder) {
                loadCollection(item.path);
            } else {
                previewFile(item.path, item.name);
            }
        });
        
        $container.append($item);
    });
}

// Update breadcrumb navigation
function updateBreadcrumb(path) {
    console.log("Updating breadcrumb for path:", path);
    
    const $breadcrumb = $('#breadcrumb');
    $breadcrumb.html('<a href="#" class="breadcrumb-item" id="breadcrumb-home">Home</a>');
    
    // Re-attach event listener
    $('#breadcrumb-home').on('click', function(e) {
        e.preventDefault();
        loadCollection('');
    });
    
    if (!path) return;
    
    const segments = path.split('/').filter(Boolean);
    let currentPath = '';
    
    segments.forEach((segment, index) => {
        currentPath += '/' + segment;
        const isLastSegment = index === segments.length - 1;
        
        if (isLastSegment) {
            $breadcrumb.append(`
                <span class="breadcrumb-separator">/</span>
                <span class="breadcrumb-item current">${segment}</span>
            `);
        } else {
            const $link = $(`
                <span class="breadcrumb-separator">/</span>
                <a href="#" class="breadcrumb-item" data-path="${currentPath}">${segment}</a>
            `);
            
            $link.on('click', function(e) {
                e.preventDefault();
                loadCollection($(this).data('path'));
            });
            
            $breadcrumb.append($link);
        }
    });
}

// Helper function to get appropriate icon class for file type
function getFileIconClass(filename) {
    const ext = filename.split('.').pop().toLowerCase();
    
    // Image files
    if (['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp', 'svg'].includes(ext)) {
        return 'fa-image';
    }
    
    // Video files
    if (['mp4', 'mov', 'avi', 'wmv', 'flv', 'mkv'].includes(ext)) {
        return 'fa-film';
    }
    
    // Audio files
    if (['mp3', 'wav', 'ogg', 'flac', 'm4a'].includes(ext)) {
        return 'fa-music';
    }
    
    // Document files
    if (['pdf', 'doc', 'docx', 'txt', 'rtf', 'odt'].includes(ext)) {
        return 'fa-file-alt';
    }
    
    // Presentation files
    if (['ppt', 'pptx', 'odp', 'key'].includes(ext)) {
        return 'fa-file-powerpoint';
    }
    
    // Spreadsheet files
    if (['xls', 'xlsx', 'ods', 'csv'].includes(ext)) {
        return 'fa-file-excel';
    }
    
    // Archive files
    if (['zip', 'rar', '7z', 'tar', 'gz'].includes(ext)) {
        return 'fa-file-archive';
    }
    
    // Code files
    if (['js', 'py', 'java', 'c', 'cpp', 'html', 'css', 'php'].includes(ext)) {
        return 'fa-file-code';
    }
    
    // Default icon for unknown file types
    return 'fa-file';
}

// Preview file
function previewFile(path, filename) {
    console.log("Previewing file:", path, filename);
    
    // Handle different file types
    const ext = filename.split('.').pop().toLowerCase();
    const isImage = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp', 'svg'].includes(ext);
    const isVideo = ['mp4', 'mov', 'avi', 'wmv'].includes(ext);
    
    if (isImage) {
        showImagePreview(path, filename);
    } else if (isVideo) {
        showVideoPreview(path, filename);
    } else {
        // For other file types, just download
        window.open(`/api/collections/download?path=${encodeURIComponent(path)}`, '_blank');
    }
}

// Image preview
function showImagePreview(path, filename) {
    // Create modal HTML
    const modal = $(`
        <div class="modal-overlay" id="image-preview-modal">
            <div class="modal-content image-preview-modal">
                <div class="modal-header">
                    <h3>${filename}</h3>
                    <button class="modal-close">&times;</button>
                </div>
                <div class="modal-body">
                    <img src="/api/collections/view?path=${encodeURIComponent(path)}" alt="${filename}">
                </div>
                <div class="modal-footer">
                    <button class="btn btn-primary" id="detect-faces-btn">
                        <i class="fas fa-user-tag"></i> Detect Faces
                    </button>
                    <button class="btn btn-light" id="download-image-btn">
                        <i class="fas fa-download"></i> Download
                    </button>
                </div>
            </div>
        </div>
    `);
    
    // Add to body
    $('body').append(modal);
    
    // Close button
    modal.find('.modal-close').on('click', function() {
        modal.remove();
    });
    
    // Download button
    modal.find('#download-image-btn').on('click', function() {
        window.open(`/api/collections/download?path=${encodeURIComponent(path)}`, '_blank');
    });
    
    // Detect faces button
    modal.find('#detect-faces-btn').on('click', function() {
        detectFaces(path, modal);
    });
    
    // Close on background click
    modal.on('click', function(e) {
        if ($(e.target).is('.modal-overlay')) {
            modal.remove();
        }
    });
}

// Detect faces in image
function detectFaces(path, modal) {
    console.log("Detecting faces in:", path);
    
    // Show loading
    const imgContainer = modal.find('.modal-body');
    const originalContent = imgContainer.html();
    imgContainer.html('<div class="loading-indicator"><div class="spinner"></div><span>Detecting faces...</span></div>');
    
    // Call face detection API
    $.ajax({
        url: '/api/detect-faces',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({ path: path }),
        success: function(response) {
            console.log("Face detection response:", response);
            
            if (response.success) {
                // Show the processed image
                imgContainer.html(`
                    <div class="face-detection-result">
                        <img src="${response.image_url}" alt="Faces detected">
                        <div class="face-count">
                            ${response.face_count} face${response.face_count !== 1 ? 's' : ''} detected
                        </div>
                    </div>
                `);
            } else {
                showToast(response.message || 'Face detection failed', 'error');
                imgContainer.html(originalContent);
            }
        },
        error: function(xhr, status, error) {
            console.error("Face detection error:", error);
            showToast('Error detecting faces', 'error');
            imgContainer.html(originalContent);
        }
    });
}

// Create a new folder
function createNewFolder() {
    const folderName = $('#new-folder-name').val().trim();
    
    if (!folderName) {
        showToast('Please enter a folder name', 'warning');
        return;
    }
    
    console.log("Creating new folder:", folderName, "in path:", currentPath);
    
    // Send request to create folder
    $.ajax({
        url: '/api/collections/create-folder',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({
            path: currentPath,
            name: folderName
        }),
        success: function(response) {
            console.log("Folder creation response:", response);
            
            if (response.success) {
                showToast('Folder created successfully', 'success');
                $('.new-folder-form').remove();
                loadCollection(currentPath); // Refresh the view
            } else {
                showToast(response.message || 'Failed to create folder', 'error');
            }
        },
        error: function(xhr, status, error) {
            console.error("Error creating folder:", error);
            showToast('Error creating folder', 'error');
        }
    });
}
