<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Stories - PhotoGenic</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css">
    <style>
        /* Modern Pro AI Stories UI */
        :root {
            --primary: #0073e6;
            --primary-light: #e6f4ff;
            --primary-dark: #0050a0;
            --secondary: #34a853;
            --accent: #fbbc05;
            --danger: #ea4335;
            --dark: #202124;
            --gray-100: #f8f9fa;
            --gray-200: #e9ecef;
            --gray-300: #dee2e6;
            --gray-400: #ced4da;
            --gray-500: #adb5bd;
            --gray-600: #6c757d;
            --gray-700: #495057;
            --gray-800: #343a40;
            --gray-900: #212529;
            --shadow-sm: 0 2px 4px rgba(0,0,0,0.05);
            --shadow: 0 4px 6px rgba(0,0,0,0.1);
            --shadow-lg: 0 10px 15px rgba(0,0,0,0.1);
            --transition: all 0.3s ease;
            --radius: 8px;
            --radius-lg: 12px;
            --font-main: 'Poppins', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            --font-serif: 'Georgia', 'Times New Roman', serif;
            --book-color: #8B4513;
            --book-spine: #6b3009;
            --book-edge: #e0d5c0;
            --page-color: #f8f5e4;
            --book-shadow: 0 15px 25px rgba(0, 0, 0, 0.2);
            --gold-foil: linear-gradient(135deg, #d4af37 0%, #f9e076 25%, #d4af37 50%, #f9e076 75%, #d4af37 100%);
        }

        body {
            font-family: var(--font-main);
            background-color: var(--gray-100);
            color: var(--gray-800);
            margin: 0;
            padding: 0;
            overflow-x: hidden;
            min-height: 100vh;
        }

        /* === Header Styling === */
        .header {
            background-color: white;
            box-shadow: var(--shadow);
            position: sticky;
            top: 0;
            z-index: 1000;
            padding: 0.8rem 1.5rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            transition: var(--transition);
        }

        .header.scrolled {
            padding: 0.5rem 1.5rem;
            box-shadow: var(--shadow-lg);
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 1.4rem;
            font-weight: 600;
            color: var(--dark);
            text-decoration: none;
            transition: var(--transition);
        }

        .logo:hover {
            color: var(--primary);
        }

        .logo i {
            color: var(--primary);
            font-size: 1.6rem;
        }

        /* === Content Area === */
        .content {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }

        /* === Button Styling === */
        .btn {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            font-family: var(--font-main);
            font-size: 0.9rem;
            font-weight: 500;
            padding: 0.6rem 1.2rem;
            border-radius: var(--radius);
            border: none;
            cursor: pointer;
            transition: var(--transition);
            white-space: nowrap;
            text-decoration: none;
        }

        .btn i {
            font-size: 1rem;
        }

        .btn-primary {
            background-color: var(--primary);
            color: white;
            box-shadow: 0 2px 4px rgba(0, 115, 230, 0.2);
        }

        .btn-primary:hover {
            background-color: var(--primary-dark);
            box-shadow: 0 4px 8px rgba(0, 115, 230, 0.3);
        }

        .btn-accent {
            background-color: var(--accent);
            color: var(--dark);
            box-shadow: 0 2px 4px rgba(251, 188, 5, 0.3);
            border: none;
        }

        .btn-accent:hover {
            background-color: #e6a800;
            box-shadow: 0 4px 8px rgba(251, 188, 5, 0.4);
        }

        .btn-light {
            background-color: white;
            color: var(--gray-700);
            border: 1px solid var(--gray-300);
        }

        .btn-light:hover {
            background-color: var(--gray-100);
            color: var(--gray-900);
        }

        /* === AI Stories UI === */
        .story-header {
            text-align: center;
            max-width: 800px;
            margin: 0 auto 2rem;
        }

        .story-title {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 1rem;
            color: var(--dark);
        }

        .story-description {
            color: var(--gray-600);
            font-size: 1.1rem;
            line-height: 1.6;
        }

        .story-container {
            background-color: white;
            border-radius: var(--radius-lg);
            box-shadow: var(--shadow-lg);
            overflow: hidden;
            margin-bottom: 2rem;
        }

        /* Form Styling */
        .story-form {
            padding: 2rem;
            background-color: white;
            border-radius: var(--radius-lg);
            box-shadow: var(--shadow);
            margin-bottom: 2rem;
        }

        .form-group {
            margin-bottom: 1.5rem;
        }

        .form-label {
            display: block;
            margin-bottom: 0.5rem;
            font-weight: 500;
            color: var(--gray-700);
        }

        .form-input {
            width: 100%;
            padding: 0.8rem 1rem;
            border: 1px solid var(--gray-300);
            border-radius: var(--radius);
            background-color: white;
            font-family: var(--font-main);
            font-size: 1rem;
            transition: var(--transition);
        }

        .form-input:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(0, 115, 230, 0.1);
        }

        .form-input::placeholder {
            color: var(--gray-500);
        }

        .form-textarea {
            height: 150px;
            resize: vertical;
        }

        /* Upload Area */
        .upload-area {
            border: 2px dashed var(--gray-300);
            border-radius: var(--radius);
            padding: 2rem;
            text-align: center;
            background-color: var(--gray-100);
            transition: var(--transition);
            cursor: pointer;
            margin-bottom: 1.5rem;
        }

        .upload-area:hover,
        .upload-area.highlight {
            border-color: var(--primary);
            background-color: var(--primary-light);
        }

        .upload-icon {
            font-size: 2.5rem;
            color: var(--gray-400);
            margin-bottom: 1rem;
            transition: var(--transition);
        }

        .upload-area:hover .upload-icon,
        .upload-area.highlight .upload-icon {
            color: var(--primary);
        }

        .upload-text {
            font-weight: 500;
            margin-bottom: 0.5rem;
            color: var(--gray-700);
        }

        .upload-subtext {
            font-size: 0.9rem;
            color: var(--gray-500);
        }

        .upload-preview {
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            margin-top: 1.5rem;
            justify-content: center;
        }

        .upload-preview-item {
            width: 100px;
            height: 100px;
            border-radius: var(--radius);
            overflow: hidden;
            position: relative;
            box-shadow: var(--shadow);
        }

        .upload-preview-item img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }

        .upload-preview-count {
            background-color: var(--primary);
            color: white;
            font-size: 0.8rem;
            font-weight: 600;
            padding: 0.25rem 0.5rem;
            border-radius: var(--radius);
            position: absolute;
            top: 0.5rem;
            right: 0.5rem;
        }

        /* Toggle switch */
        .toggle-container {
            display: flex;
            align-items: center;
            margin-bottom: 1.5rem;
            gap: 1rem;
            background-color: var(--gray-100);
            border-radius: var(--radius-lg);
            padding: 0.5rem;
        }

        .toggle-option {
            flex: 1;
            padding: 0.8rem 1rem;
            text-align: center;
            border-radius: var(--radius);
            cursor: pointer;
            transition: var(--transition);
            font-weight: 500;
            color: var(--gray-600);
        }

        .toggle-option.active {
            background-color: white;
            color: var(--primary);
            box-shadow: var(--shadow-sm);
        }

        .toggle-option i {
            margin-right: 0.5rem;
        }

        /* Results area styling */
        .results-area {
            display: none;
            margin-top: 2rem;
        }

        .results-area.active {
            display: block;
        }

        /* 3D Book Styling - Enhanced version */
        .book-container {
            display: flex;
            align-items: center;
            justify-content: center;
            perspective: 1200px;
            width: 100%;
            margin: 2rem auto;
            height: 450px;
            position: relative;
        }

        @keyframes initAnimation {
            0% {
                transform: rotateY(0deg);
            }
            100% {
                transform: rotateY(-30deg);
            }
        }

        .book-3d {
            width: 280px;
            height: 400px;
            position: relative;
            transform-style: preserve-3d;
            transform: rotateY(-30deg);
            transition: transform 1.2s cubic-bezier(0.645, 0.045, 0.355, 1);
            animation: 1s ease 0s 1 initAnimation;
            cursor: pointer;
            box-shadow: var(--book-shadow);
        }

        .book-3d.book-open {
            transform: rotateY(0deg);
        }

        .book-cover {
            position: absolute;
            top: 0;
            left: 0;
            width: 280px;
            height: 400px;
            transform: translateZ(20px);
            background: linear-gradient(145deg, var(--book-color) 0%, #9c6a44 60%, #8B4513 100%);
            border-radius: 0 4px 4px 0;
            box-shadow: 
                5px 5px 20px rgba(0, 0, 0, 0.3),
                inset 0 0 50px rgba(0, 0, 0, 0.2);
            backface-visibility: hidden;
            transition: transform 1.5s cubic-bezier(0.645, 0.045, 0.355, 1);
            transform-origin: 0% 50%;
            overflow: hidden;
            z-index: 10;
        }

        .book-cover.open {
            transform: rotateY(-180deg) translateZ(20px);
            box-shadow: 
                -5px 5px 20px rgba(0, 0, 0, 0.2),
                inset 0 0 50px rgba(0, 0, 0, 0.3);
        }

        .book-spine {
            position: absolute;
            left: 0;
            top: 0;
            width: 40px;
            height: 400px;
            transform: translateX(240px) rotateY(90deg);
            background: linear-gradient(90deg,
                var(--book-spine) 0%,
                #8B5A2B 40%,
                #A67C52 50%,
                #8B5A2B 60%,
                var(--book-spine) 100%);
            box-shadow: inset 0px 0px 5px rgba(0, 0, 0, 0.2);
            display: flex;
            align-items: center;
            justify-content: center;
            writing-mode: vertical-rl;
            text-orientation: mixed;
            color: rgba(255, 255, 255, 0.8);
            font-family: var(--font-serif);
            font-size: 1.2rem;
            padding: 10px 5px;
            overflow: hidden;
            text-overflow: ellipsis;
            z-index: 5;
        }

        .book-back {
            position: absolute;
            top: 0;
            left: 0;
            width: 280px;
            height: 400px;
            transform: translateZ(-20px) rotateY(180deg);
            background-color: var(--book-color);
            border-radius: 4px 0 0 4px;
            box-shadow: -10px 0 50px 10px rgba(0, 0, 0, 0.2);
            z-index: 0;
        }

        .book-edges {
            position: absolute;
            top: 0;
            left: 0;
            width: 280px;
            height: 400px;
            transform: translateX(12px) rotateY(90deg);
            transform-origin: left;
            background-color: var(--book-edge);
            box-shadow: inset 0 0 5px rgba(0, 0, 0, 0.1);
            z-index: 4;
        }

        .book-edges::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: repeating-linear-gradient(
                to right, 
                transparent, 
                transparent 3px, 
                rgba(0, 0, 0, 0.05) 3px,
                rgba(0, 0, 0, 0.05) 4px
            );
        }

        .book-pages {
            position: absolute;
            top: 5px;
            left: 5px;
            width: 270px;
            height: 390px;
            transform: translateZ(10px);
            background-color: var(--page-color);
            box-shadow: 0 0 3px rgba(0, 0, 0, 0.1);
            display: none;
            overflow: hidden;
            z-index: 5;
        }

        .book-3d.book-open .book-pages {
            display: block;
        }

        .book-page {
            position: absolute;
            width: 100%;
            height: 100%;
            top: 0;
            left: 0;
            padding: 15px;
            box-sizing: border-box;
            font-family: var(--font-serif);
            font-size: 0.9rem;
            line-height: 1.5;
            color: #333;
            background-color: var(--page-color);
            background-image:
                linear-gradient(90deg, rgba(0,0,0,0.03) 0%, transparent 10%, transparent 90%, rgba(0,0,0,0.03) 100%);
            overflow: hidden;
        }

        .book-page::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image:
                linear-gradient(var(--page-color) 0.8px, transparent 0.9 px),
                linear-gradient(90deg, #e0decd 0.8 px, transparent 0.9 px);
            background-size: 20px 20px;
            opacity: 0.07;
            pointer-events: none;
        }

        .page-content {
            height: 100%;
            overflow: hidden;
        }

        .book-page-spread {
            display: none;
            width: 100%;
            height: 100%;
        }

        .book-page-spread.active {
            display: flex;
        }

        .page-left, .page-right {
            width: 50%;
            height: 100%;
            padding: 15px;
            box-sizing: border-box;
            overflow: hidden;
        }

        .page-left {
            border-right: 1px solid rgba(0,0,0,0.1);
            box-shadow: 5px 0 5px -5px rgba(0,0,0,0.1);
        }

        .cover-content {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100%;
            width: 100%;
            padding: 30px;
            box-sizing: border-box;
            text-align: center;
            color: #fff;
            background: linear-gradient(135deg, var(--book-color) 0%, #9c6a44 100%);
            position: relative;
            overflow: hidden;
        }

        .cover-content::before {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 80%;
            height: 80%;
            transform: translate(-50%, -50%);
            border: 3px solid var(--gold-foil);
            box-shadow: 0 0 10px rgba(212, 175, 55, 0.3);
            pointer-events: none;
            opacity: 0.7;
        }

        .cover-decoration {
            position: absolute;
            width: 60px;
            height: 60px;
            background: var(--gold-foil);
            opacity: 0.8;
            mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M12 1L9 9H2L7 14.5L5 22L12 17.5L19 22L17 14.5L22 9H15L12 1Z'/%3E%3C/svg%3E");
            -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M12 1L9 9H2L7 14.5L5 22L12 17.5L19 22L17 14.5L22 9H15L12 1Z'/%3E%3C/svg%3E");
            mask-size: contain;
            -webkit-mask-size: contain;
            mask-repeat: no-repeat;
            -webkit-mask-repeat: no-repeat;
        }

        .cover-decoration:nth-child(1) {
            top: 20px;
            left: 20px;
        }

        .cover-decoration:nth-child(2) {
            top: 20px;
            right: 20px;
        }

        .cover-decoration:nth-child(3) {
            bottom: 20px;
            left: 20px;
        }

        .cover-decoration:nth-child(4) {
            bottom: 20px;
            right: 20px;
        }

        .page-turn-overlay {
            position: absolute;
            width: 120px;
            height: 400px;
            top: 0;
            right: 0;
            background-image: linear- gradient(to right, transparent 0%, rgba(0,0,0,0.01) 100%);
            transform: translateZ(25px);
            z-index: 20;
            display: none;
            cursor: pointer;
        }

        .book-3d.book-open .page-turn-overlay {
            display: block;
        }

        .page-turn-animation {
            position: absolute;
            top: 0;
            right: 0;
            width: 270px;
            height: 390px;
            background-color: var(--page-color);
            box-shadow: -5px 0 10px rgba(0,0,0,0.1);
            transform-origin: left center;
            transform-style: preserve-3d;
            transition: transform 0.8s ease;
            z-index: 15;
            display: none;
        }

        .book-navigation {
            display: flex;
            align-items: center;
            justify-content: space-between;
            width: 100%;
            max-width: 400px;
            margin: 1rem auto;
        }

        .book-nav-btn {
            background-color: var(--color-accent);
            color: white;
            border: none;
            border-radius: 20px;
            padding: 8px 16px;
            font-size: 0.9rem;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 5px;
            transition: all 0.3s ease;
        }

        .book-nav-btn:hover:not(:disabled) {
            background-color: var(--color-accent-dark);
            transform: translateY(-2px);
        }

        .book-nav-btn:disabled {
            background-color: #ccc;
            cursor: not-allowed;
            transform: none;
        }

        .book-nav-btn i {
            font-size: 0.8rem;
        }

        @keyframes turnPage {
            0% {
                transform: rotateY(0deg);
            }
            100% {
                transform: rotateY(-180deg);
            }
        }

        .turning {
            animation: turnPage 0.8 s ease forwards;
        }

        @media (max-width: 768px) {
            .book-3d {
                width: 200px;
                height: 300px;
            }

            .book-cover, .book-back {
                width: 200px;
                height: 300px;
            }

            .book-spine {
                height: 300px;
                transform: translateX(180px) rotateY(90deg);
            }

            .book-pages {
                width: 190px;
                height: 290px;
            }

            .page-turn-overlay {
                height: 300px;
            }

            .cover-content h1 {
                font-size: 1.6rem;
            }

            .cover-content h2 {
                font-size: 1rem;
            }
        }

        /* Book controls */
        .book-controls {
            display: flex;
            justify-content: center;
            gap: 1rem;
            margin-top: 2rem;
        }

        .control-btn {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 50%;
            width: 45px;
            height: 45px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            font-size: 1rem;
            color: #495057;
            transition: all 0.2s ease;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }

        .control-btn:hover {
            background-color: #e9ecef;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
            transform: translateY(-2px);
        }

        .control-btn:active {
            transform: translateY(0);
            box-shadow: 0 2px 3px rgba(0, 0, 0, 0.1);
        }

        .control-btn.disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        /* Navigation */
        .book-navigation {
            display: flex;
            justify-content: space-between;
            width: 100%;
            margin-top: 1rem;
        }

        .nav-btn {
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 0.75rem 1.5rem;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-weight: 500;
            transition: background-color 0.2 s, transform 0.2s;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }

        .nav-btn:hover:not(:disabled) {
            background-color: #0069d9;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
        }

        .nav-btn:active:not(:disabled) {
            transform: translateY(0);
        }

        .nav-btn:disabled {
            background-color: #6c757d;
            cursor: not-allowed;
            opacity: 0.65;
        }

        .nav-btn i {
            margin: 0 0.5rem;
        }

        /* Page content styling */
        .page-content {
            font-family: var(--font-serif);
            line-height: 1.8;
            color: #333;
        }

        .page-content h1, .page-content h2 {
            font-family: var(--font-serif);
            margin-bottom: 1.5rem;
            color: #1c3968;
        }

        .page-content p {
            margin-bottom: 1rem;
            text-align: justify;
            text-indent: 2rem;
        }

        .page-content p:first-of-type:first-letter {
            initial-letter: 3;
            -webkit-initial-letter: 3;
            margin-right: 0.5rem;
            color: #1c3968;
            font-family: var(--font-serif);
            font-weight: bold;
            font-size: 3rem;
            float: left;
            line-height: 1;
        }

        .page-image {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 1rem auto 2rem;
            border-radius: 5px;
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.2);
            transition: transform 0.3 s ease;
        }

        .page-image:hover {
            transform: scale(1.02);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
        }

        /* Book cover title */
        .book-title {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 1rem;
            font-family: var(--font-serif);
            text-shadow: 0 2px 5px rgba(0, 0, 0, 0.5);
            letter-spacing: 1px;
            background: linear-gradient(45deg, #fff 10%, #ccc 90%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-fill-color: transparent;
        }

        .book-subtitle {
            font-size: 1.2rem;
            margin-bottom: 2rem;
            font-family: var(--font-serif);
            font-style: italic;
            text-shadow: 0 1px 3px rgba(0, 0, 0, 0.5);
            opacity: 0.9;
        }

        .read-btn {
            padding: 0.75rem 1.5rem;
            margin-top: 2rem;
            font-weight: 600;
            transition: transform 0.3s, box-shadow 0.3s;
            background-color: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(5px);
            border: 2px solid rgba(255, 255, 255, 0.2);
            color: white;
            border-radius: 30px;
            font-size: 1rem;
            letter-spacing: 1px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            cursor: pointer;
        }

        .read-btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
            background-color: rgba(255, 255, 255, 0.25);
        }

        .read-btn:active {
            transform: translateY(-1px);
        }

        /* Page counter */
        .page-counter {
            font-family: var(--font-serif);
            text-align: center;
            margin-top: 1rem;
            font-style: italic;
            color: #6c757d;
        }

        /* Loading spinner overlay */
        .loading-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: rgba(255,255,255,0.8);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 1000;
            opacity: 0;
            visibility: hidden;
            transition: opacity 0.3 s, visibility 0.3 s;
        }

        .loading-overlay.active {
            opacity: 1;
            visibility: visible;
        }

        .spinner {
            display: inline-block;
            width: 80px;
            height: 80px;
            border: 8px solid var(--primary-light);
            border-radius: 50%;
            border-top-color: var(--primary);
            animation: spin 1s ease-in-out infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* Form buttons */
        .form-btn {
            padding: 0.75rem 1.5rem;
            border-radius: var(--radius);
            border: none;
            font-weight: 600;
            cursor: pointer;
            transition: var(--transition);
            font-size: 1rem;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
        }

        .btn-primary {
            background-color: var(--primary);
            color: white;
        }

        .btn-primary:hover {
            background-color: var(--primary-dark);
        }

        .book-nav-btn.disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .book-navigation {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 1rem;
            margin-top: 1rem;
        }

        .book-nav-btn {
            background-color: var(--primary);
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: var(--radius);
            cursor: pointer;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .page-counter {
            font-weight: 500;
            font-size: 1.1rem;
        }

        /* Story action buttons styling */
        .story-actions {
            display: flex;
            gap: 15px;
            margin-top: 25px;
            justify-content: center;
        }

        .story-action-btn {
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 12px 20px;
            border-radius: var(--radius);
            border: none;
            cursor: pointer;
            transition: var(--transition);
            font-weight: 500;
            overflow: hidden;
            min-width: 180px;
            color: white;
            box-shadow: var(--shadow);
        }

        .story-action-btn i {
            margin-right: 10px;
            font-size: 1.2rem;
        }

        .story-action-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        /* Button hover effects */
        .story-action-btn:not(:disabled):hover {
            transform: translateY(-3px);
            box-shadow: var(--shadow-lg);
        }

        .story-action-btn:not(:disabled):active {
            transform: translateY(-1px);
        }

        /* Button types */
        .download-btn {
            background: linear-gradient(135deg, #4CAF50, #2E7D32);
        }

        .save-btn {
            background: linear-gradient(135deg, #FF4081, #C2185B);
        }

        .share-btn {
            background: linear-gradient(135deg, #2196F3, #0D47A1);
        }

        /* Button ripple effect */
        .story-action-btn .ripple {
            position: absolute;
            border-radius: 50%;
            background-color: rgba(255, 255, 255, 0.4);
            transform: scale(0);
            animation: ripple 0.6s linear;
            pointer-events: none;
        }

        @keyframes ripple {
            to {
                transform: scale(4);
                opacity: 0;
            }
        }

        /* Button tooltip */
        .action-tooltip {
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%) translateY(-5px);
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 5px 10px;
            border-radius: 4px;
            font-size: 12px;
            white-space: nowrap;
            opacity: 0;
            visibility: hidden;
            transition: all 0.3s ease;
            z-index: 10;
        }

        .story-action-btn:hover .action-tooltip {
            opacity: 1;
            visibility: visible;
            transform: translateX(-50%) translateY(-10px);
        }

        /* Success animation */
        .story-action-btn.success {
            background: #4CAF50;
            animation: success-pulse 1 s;
        }

        @keyframes success-pulse {
            0% { box-shadow: 0 0 0 0 rgba(76, 175, 80, 0.7); }
            70% { box-shadow: 0 0 0 10 px rgba(76, 175, 80, 0); }
            100% { box-shadow: 0 0 0 0 rgba(76, 175, 80, 0); }
        }

        .cover-content h1 {
            font-family: var(--font-serif);
            font-size: 2rem;
            margin-bottom: 1rem;
            text-shadow: 2px 2px 3px rgba(0,0,0,0.3);
            max-width: 100%;
            word-wrap: break-word;
            position: relative;
            z-index: 5;
        }

        .cover-content h2 {
            font-family: var(--font-serif);
            font-style: italic;
            font-weight: normal;
            font-size: 1.2rem;
            margin-top: 0;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
            opacity: 0.9;
            max-width: 100%;
            word-wrap: break-word;
            position: relative;
            z-index: 5;
        }

        .book-cover::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-image: url('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADIAAAAyCAMAAAAp4XiDAAAAUVBMVEWFhYWDg4N3d3dtbW17e3t1dXWBgYGHh4d5eXlzc3OLi4ubm5uVlZWPj4+NjY19fX2JiYl/f39ra2uRkZGZmZlpaWmXl5dvb29xcXGTk5NnZ2c8TV1mAAAAG3RSTlNAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEAvEOwtAAAFVklEQVR4XpWWB67c2BUFb3g557T/hRo9/WUMZHlgr4Bg8Z4qQgQJlHI4A8SzFVrapvmTF9O7dmYRFZ60YiBhJRCgh1FYhiLAmdvX0CzTOpNE77ME0Zty/nWWzchDtiqrmQDeuv3powQ5ta2eN0FY0InkqDD73lT9c9lEzwUNqgFHs9VQce3TVClFCQrSTfOiYkVJQBmpbq2L6iZavPnAPcoU0dSw0SUTqz/GtrGuXfbyyBniKykOWQWGqwwMA7QiYAxi+IlPdqo+hYHnUt5ZPfnsHJyNiDtnpJyayNBkF6cWoYGAMY92U2hXHF/C1M8uP/ZtYdiuj26UdAdQQSXQErwSOMzt/XWRWAz5GuSBIkwG1H3FabJ2OsUOUhGC6tK4EMtJO0ttC6IBD3kM0ve0tJwMdSfjZo+EEISaeTr9P3wYrGjXqyC1krcKdhMpxEnt5JetoulscpyzhXN5FRpuPHvbeQaKxFAEB6EN+cYN6xD7RYGpXpNndMmZgM5Dcs3YSNFDHUo2LGfZuukSWyUYirJAdYbF3MfqEKmjM+I2EfhA94iG3L7uKrR+GdWD73ydlIB+6hgref1QTlmgmbM3/LeX5GI1Ux1RWpgxpLuZ2+I+IjzZ8wqE4nilvQdkUdfhzI5QDWy+kw5Wgg2pGpeEVeCCA7b85BO3F9DzxB3cdqvBzWcmzbyMiqhzuYqtHRVG2y4x+KOlnyqla8AoWWpuBoYRxzXrfKuILl6SfiWCbjxoZJUaCBj1CjH7GIaDbc9kqBY3W/Rgjda1iqQcOJu2WW+76pZC9QG7M00dffe9hNnseupFL53r8F7YHSwJWUKP2q+k7RdsxyOB11n0xtOvnW4irMMFNV4H0uqwS5ExsmP9AxbDTc9JwgneAT5vTiUSm1E7BSflSt3bfa1tv8Di3R8n3Af7MNWzs49hmauE2wP+ttrq+AsWpFG2awvsuOqbipWHgtuvuaAE+A1Z/7gC9hesnr+7wqCwG8c5yAg3AL1fm8T9AZtp/bbJGwl1pNrE7RuOX7PeMRUERVaPpEs+yqeoSmuOlokqw49pgomjLeh7icHNlG19yjs6XXOMedYm5xH2YxpV2tc0Ro2jJfxC50ApuxGob7lMsxfTbeUv07TyYxpeLucEH1gNd4IKH2LAg5TdVhlCafZvpskfncCfx8pOhJzd76bJWeYFnFciwcYfubRc12Ip/ppIhA1/mSZ/RxjFDrJC5xifFjJpY2Xl5zXdguFqYyTR1zSp1Y9p+tktDYYSNflcxI0iyO4TPBdlRcpeqjK/piF5bklq77VSEaA+z8qmJTFzIWiitbnzR794USKBUaT0NTEsVjZqLaFVqJoPN9ODG70IPbfBHKK+/q/AWR0tJzYHRULOa4MP+W/HfGadZUbfw177G7j/OGbIs8TahLyynl4X4RinF793Oz+BU0saXtUHrVBFT/DnA3ctNPoGbs4hRIjTok8i+algT1lTHi4SxFvONKNrgQFAq2/gFnWMXgwffgYMJpiKYkmW3tTg3ZQ9Jq+f8XN+A5eeUKHWvJWJ2sgJ1Sop+wwhqFVijqWaJhwtD8MNlSBeWNNWTa5Z5kPZw5+LbVT99wqTdx29lMUH4OIG/D86ruKEauBjvH5xy6um/Sfj7ei6UUVk4AIl3MyD4MSSTOFgSwsH/QJWaQ5as7ZcmgBZkzjjU1UrQ74ci1gWBCSGHtuV1H2mhSnO3Wp/3fEV5a+4wz//6qy8JxjZsmxxy5+4w9CDNJY09T072iKG0EnOS0arEYgXqYnXcYHwjTtUNAcMelOd4xpkoqiTYICWFq0JSiPfPDQdnt+4/wuqcXY47QILbgAAAABJRU5ErkJggg==');
            opacity: 0.1;
            pointer-events: none;
        }

        @keyframes pageTurning {
            0% { transform: rotateY(0deg); }
            100% { transform: rotateY(-180deg); }
        }

        @keyframes pageAppearsFromLeft {
            0% { opacity: 0; transform: translateX(-20px); }
            100% { opacity: 1; transform: translateX(0); }
        }

        @keyframes pageDisappearsRight {
            0% { opacity: 1; transform: translateX(0); }
            100% { opacity: 0; transform: translateX(20px); }
        }

        @keyframes softBounce {
            0% { transform: rotateY(0deg) translateY(0); }
            50% { transform: rotateY(0deg) translateY(-10px); }
            100% { transform: rotateY(0deg) translateY(0); }
        }
    </style>
</head>
<body>
    <!-- Pro Header with Modern UI -->
    <header class="header">
        <a href="/" class="logo">
            <i class="fas fa-camera"></i>
            <span>PhotoGenic</span>
        </a>
        <div class="action-bar">
            <a href="/collections" class="btn btn-light">
                <i class="fas fa-photo-video"></i> Collections
            </a>
            <a href="/ai-features" class="btn btn-accent">
                <i class="fas fa-robot"></i> AI Features
            </a>
        </div>
    </header>

    <!-- Main Content -->
    <main class="content">
        <div class="story-header">
            <h1 class="story-title">Create AI Stories</h1>
            <p class="story-description">Turn your photos or text into captivating stories with AI. Generate unique narratives that capture the essence of your memories.</p>
        </div>

        <div class="story-container">
            <form id="story-form" class="story-form" onsubmit="handleGenerateStory(event)">
                <div class="toggle-container">
                    <div class="toggle-option active" id="toggle-image" onclick="setMode('image')">
                        <i class="fas fa-image"></i> Generate from Image
                    </div>
                    <div class="toggle-option" id="toggle-text" onclick="setMode('text')">
                        <i class="fas fa-font"></i> Generate from Text
                    </div>
                </div>

                <div class="form-group">
                    <label for="story-title" class="form-label">Story Title</label>
                    <input type="text" id="story-title" class="form-input" placeholder="Enter a title for your story" required>
                </div>

                <div class="form-group">
                    <label for="story-subtitle" class="form-label">Story Subtitle (optional)</label>
                    <input type="text" id="story-subtitle" class="form-input" placeholder="Enter a subtitle for your story">
                </div>

                <div class="form-group" id="image-upload-group">
                    <label class="form-label">Upload Images</label>
                    <div class="upload-area" id="image-upload-area">
                        <div class="upload-icon">
                            <i class="fas fa-cloud-upload-alt"></i>
                        </div>
                        <div class="upload-text">
                            Drag &amp; drop your images here or
                            <button type="button" class="browse-btn" id="browse-btn">Browse Files</button>
                            <input type="file" id="file-input" accept="image/*" multiple style="display: none;">
                        </div>
                        <div class="upload-instructions">
                            You can upload up to 5 images. Supported formats: JPG, PNG, GIF
                        </div>
                    </div>
                    <div class="upload-preview" id="upload-preview" style="display: flex; flex-wrap: wrap; gap: 1rem;"></div>
                    <div class="progress-container" id="progress-container" style="display: none;">
                        <div class="progress-bar" id="progress-bar"></div>
                        <div class="progress-text">Uploading...</div>
                    </div>
                </div>

                <div class="form-group" id="text-prompt-group" style="display: none;">
                    <label class="form-label" for="text-prompt">Describe Your Story</label>
                    <textarea id="text-prompt" class="form-input form-textarea" rows="5" placeholder="Describe what you want your story to be about..."></textarea>
                    <div class="form-hint">
                        For best results, include details about characters, settings, themes, and any specific plot elements.
                    </div>
                </div>

                <div class="form-group">
                    <label class="form-label">Story Style</label>
                    <div class="style-options">
                        <label class="style-option">
                            <input type="radio" name="style" value="fantasy" checked>
                            <span class="style-name">Fantasy</span>
                        </label>
                        <label class="style-option">
                            <input type="radio" name="style" value="mystery">
                            <span class="style-name">Mystery</span>
                        </label>
                        <label class="style-option">
                            <input type="radio" name="style" value="romance">
                            <span class="style-name">Romance</span>
                        </label>
                        <label class="style-option">
                            <input type="radio" name="style" value="adventure">
                            <span class="style-name">Adventure</span>
                        </label>
                        <label class="style-option">
                            <input type="radio" name="style" value="scifi">
                            <span class="style-name">Sci-Fi</span>
                        </label>
                    </div>
                </div>

                <div class="form-actions">
                    <button type="submit" class="form-btn btn-primary" id="generate-btn">
                        <i class="fas fa-wand-magic-sparkles"></i> Generate Story
                    </button>
                    <button type="button" class="form-btn btn-outline" id="reset-btn" onclick="resetForm()">
                        <i class="fas fa-redo"></i> Reset
                    </button>
                </div>
            </form>
        </div>

        <div class="results-area" id="story-result">
            <div class="book-container">
                <div id="book-3d" class="book-3d book-closed">
                    <!-- Book cover -->
                    <div id="book-cover" class="book-cover closed">
                        <div class="cover-decoration"></div>
                        <div class="cover-decoration"></div>
                        <div class="cover-decoration"></div>
                        <div class="cover-decoration"></div>
                        <div class="cover-content">
                            <h1 id="book-title">Your Story</h1>
                            <h2 id="book-subtitle">A Tale of Imagination</h2>
                        </div>
                    </div>

                    <!-- Book spine -->
                    <div class="book-spine" id="book-spine">Your Story</div>

                    <!-- Book edges -->
                    <div class="book-edges" id="book-edges"></div>

                    <!-- Book back cover -->
                    <div class="book-back"></div>

                    <!-- Book pages -->
                    <div class="book-pages" id="book-pages">
                        <!-- Pages will be dynamically added here -->
                    </div>

                    <!-- Page turn overlay -->
                    <div class="page-turn-overlay" id="page-turn-overlay"></div>
                    <div class="page-turn-animation" id="page-turn-animation"></div>
                </div>
            </div>

            <!-- Book navigation -->
            <div class="book-navigation" id="book-navigation">
                <button id="prev-page-btn" class="book-nav-btn" disabled>
                    <i class="fas fa-arrow-left"></i> Previous
                </button>
                <div class="page-counter">
                    Page <span id="current-page">1</span> of <span id="total-pages">1</span>
                </div>
                <button id="next-page-btn" class="book-nav-btn" disabled>
                    Next <i class="fas fa-arrow-right"></i>
                </button>
            </div>

            <!-- Read button -->
            <div class="text-center my-3">
                <button id="read-btn" class="btn btn-primary" onclick="openBook()">
                    <i class="fas fa-book-open"></i> Open Book
                </button>
            </div>
        </div>
    </main>

    <!-- Loading Overlay -->
    <div class="loading-overlay" id="loading-overlay">
        <div class="spinner"></div>
    </div>

    <!-- Toast Container -->
    <div id="toast-container"></div>

    <script src="https://kit.fontawesome.com/a076d05399.js" crossorigin="anonymous"></script>
    <script>
        // Global variables for DOM elements - these will be initialized when DOM is loaded
        let storyForm = null;
        let resetBtn = null;
        let generateBtn = null;
        let fileInput = null;
        let browseBtn = null;
        let uploadArea = null;
        let uploadPreview = null;
        let imageUploadGroup = null;
        let textPromptGroup = null;
        let toggleImage = null;
        let toggleText = null;
        let loadingOverlay = null;
        let progressContainer = null;
        let progressBar = null;

        // Global variables
        let isImageMode = true;
        let uploadedImages = [];

        // Initialize the page when DOM is loaded
        document.addEventListener('DOMContentLoaded', function() {
            console.log('Initializing AI Stories page');

            // Initialize DOM elements - do this once at the beginning
            storyForm = document.getElementById('story-form');
            resetBtn = document.getElementById('reset-btn');
            generateBtn = document.getElementById('generate-btn');
            fileInput = document.getElementById('file-input');
            browseBtn = document.getElementById('browse-btn');
            uploadArea = document.getElementById('image-upload-area');
            uploadPreview = document.getElementById('upload-preview');
            imageUploadGroup = document.getElementById('image-upload-group');
            textPromptGroup = document.getElementById('text-prompt-group');
            toggleImage = document.getElementById('toggle-image');
            toggleText = document.getElementById('toggle-text');
            loadingOverlay = document.getElementById('loading-overlay');
            progressContainer = document.getElementById('progress-container');
            progressBar = document.getElementById('progress-bar');

            // Log all initialized elements for debugging
            console.log('DOM Elements initialized:', {
                storyForm: !!storyForm,
                resetBtn: !!resetBtn,
                generateBtn: !!generateBtn,
                fileInput: !!fileInput,
                browseBtn: !!browseBtn,
                uploadArea: !!uploadArea,
                uploadPreview: !!uploadPreview,
                imageUploadGroup: !!imageUploadGroup,
                textPromptGroup: !!textPromptGroup,
                toggleImage: !!toggleImage,
                toggleText: !!toggleText,
                loadingOverlay: !!loadingOverlay
            });

            // Initialize reset button
            if (resetBtn) {
                resetBtn.addEventListener('click', resetForm);
                console.log('Reset button event handler attached');
            }

            // Add form submit handler (use single binding point)
            if (storyForm) {
                storyForm.addEventListener('submit', function(event) {
                    event.preventDefault();
                    console.log('Form submitted');
                    handleGenerateStory(event);
                });
                console.log('Form submit handler attached');
            }

            // Initialize file input
            if (browseBtn && fileInput) {
                browseBtn.addEventListener('click', function() {
                    console.log('Browse button clicked');
                    fileInput.click();
                });

                fileInput.addEventListener('change', function() {
                    console.log('File input changed');
                    handleFiles(this.files);
                });
                console.log('File input handlers attached');
            }

            // Initialize drag and drop for image upload
            if (uploadArea) {
                ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(event => {
                    uploadArea.addEventListener(event, function(e) {
                        e.preventDefault();
                        e.stopPropagation();
                    });
                });

                ['dragenter', 'dragover'].forEach(event => {
                    uploadArea.addEventListener(event, function() {
                        this.classList.add('highlight');
                    });
                });

                ['dragleave', 'drop'].forEach(event => {
                    uploadArea.addEventListener(event, function() {
                        this.classList.remove('highlight');
                    });
                });

                uploadArea.addEventListener('drop', function(e) {
                    console.log('Files dropped');
                    const files = e.dataTransfer.files;
                    handleFiles(files);
                });
                console.log('Drag and drop handlers attached');
            }

            // Set default mode
            console.log('Setting default mode to image');
            setMode('image');

            // Initialize any empty image preview containers
            showImagePreviews();

            // Initialize the book navigation
            initBookNavigation();

            console.log('AI Stories initialization complete');
        });

        // Set current mode (image or text)
        function setMode(mode) {
            console.log('Setting mode to:', mode);
            isImageMode = mode === 'image';

            // Ensure we have references to the toggle elements
            if (!toggleImage) toggleImage = document.getElementById('toggle-image');
            if (!toggleText) toggleText = document.getElementById('toggle-text');
            if (!imageUploadGroup) imageUploadGroup = document.getElementById('image-upload-group');
            if (!textPromptGroup) textPromptGroup = document.getElementById('text-prompt-group');

            // Update toggle buttons
            if (toggleImage) {
                toggleImage.classList.toggle('active', isImageMode);
            } else {
                console.error('Toggle image element not found');
            }

            if (toggleText) {
                toggleText.classList.toggle('active', !isImageMode);
            } else {
                console.error('Toggle text element not found');
            }

            // Show/hide appropriate input fields
            if (imageUploadGroup) {
                imageUploadGroup.style.display = isImageMode ? 'block' : 'none';
            } else {
                console.error('Image upload group element not found');
            }

            if (textPromptGroup) {
                textPromptGroup.style.display = isImageMode ? 'none' : 'block';
            } else {
                console.error('Text prompt group element not found');
            }

            console.log('Mode switched to:', isImageMode ? 'image' : 'text');
        }

        // Initialize book navigation
        function initBookNavigation() {
            const prevBtn = document.getElementById('prev-page-btn');
            const nextBtn = document.getElementById('next-page-btn');

            if (prevBtn) {
                prevBtn.addEventListener('click', function() {
                    console.log('Previous page button clicked');
                    goToPage('prev');
                });
            }

            if (nextBtn) {
                nextBtn.addEventListener('click', function() {
                    console.log('Next page button clicked');
                    goToPage('next');
                });
            }

            console.log('Book navigation initialized');
        }

        // Update page navigation
        function updatePageNavigation(totalSpreads) {
            console.log('Updating page navigation for', totalSpreads, 'spreads');

            const prevBtn = document.getElementById('prev-page-btn');
            const nextBtn = document.getElementById('next-page-btn');
            const currentPage = document.getElementById('current-page');
            const totalPages = document.getElementById('total-pages');

            if (totalPages) {
                totalPages.textContent = totalSpreads;
            }

            if (currentPage) {
                currentPage.textContent = '1';
            }

            if (prevBtn) {
                prevBtn.disabled = true;
            }

            if (nextBtn) {
                nextBtn.disabled = totalSpreads <= 1;
            }

            // Attach click events to navigation buttons
            if (prevBtn && !prevBtn.hasAttribute('data-initialized')) {
                prevBtn.setAttribute('data-initialized', 'true');
                prevBtn.addEventListener('click', function() {
                    goToPage('prev');
                });
            }

            if (nextBtn && !nextBtn.hasAttribute('data-initialized')) {
                nextBtn.setAttribute('data-initialized', 'true');
                nextBtn.addEventListener('click', function() {
                    goToPage('next');
                });
            }
        }

        // Go to specific book page
        function goToPage(direction) {
            console.log('Going to page direction:', direction);

            const currentPage = document.getElementById('current-page');
            const totalPages = document.getElementById('total-pages');
            const prevBtn = document.getElementById('prev-page-btn');
            const nextBtn = document.getElementById('next-page-btn');

            if (!currentPage || !totalPages) {
                console.error('Page navigation elements not found');
                return;
            }

            let current = parseInt(currentPage.textContent);
            const total = parseInt(totalPages.textContent);

            if (isNaN(current) || isNaN(total)) {
                console.error('Invalid page numbers:', current, total);
                return;
            }

            // Determine the target page
            let target = current;
            if (direction === 'prev' && current > 1) {
                target = current - 1;
            } else if (direction === 'next' && current < total) {
                target = current + 1;
            } else if (typeof direction === 'number' && direction >= 1 && direction <= total) {
                target = direction;
            }

            if (target === current) {
                console.log('Already on page', target);
                return;
            }

            console.log('Navigating from page', current, 'to page', target);

            // Hide current spread and show target spread
            const spreads = document.querySelectorAll('.book-page-spread');

            if (spreads.length === 0) {
                console.error('No book spreads found');
                return;
            }

            for (let i = 0; i < spreads.length; i++) {
                if (i === target - 1) {
                    spreads[i].style.display = 'flex';
                } else {
                    spreads[i].style.display = 'none';
                }
            }

            // Update page indicator
            currentPage.textContent = target;

            // Update button states
            if (prevBtn) {
                prevBtn.disabled = target === 1;
            }

            if (nextBtn) {
                nextBtn.disabled = target === total;
            }
        }

        // Reset the form
        function resetForm() {
            console.log('Resetting form');

            // Reset form fields
            const storyForm = document.getElementById('story-form');
            if (storyForm) {
                storyForm.reset();
            }

            // Clear image previews
            uploadedImages = [];
            const uploadPreview = document.getElementById('upload-preview');
            if (uploadPreview) {
                uploadPreview.innerHTML = '';
            }

            // Hide results
            const storyResult = document.getElementById('story-result');
            if (storyResult) {
                storyResult.style.display = 'none';
            }

            showToast('Form reset successfully', 'info');
        }

        // Handle file uploads
        function handleFiles(files) {
            console.log('Handling files:', files?.length || 0);

            if (!files || files.length === 0) {
                console.log('No files to process');
                return;
            }

            // Switch to image mode since we're uploading images
            setMode('image');

            // Filter valid image files and limit to 5
            const fileArray = Array.from(files)
                .filter(file => file.type.startsWith('image/'))
                .slice(0, 5);

            if (fileArray.length === 0) {
                showToast('Please upload image files only (JPG, PNG, GIF)', 'error');
                return;
            }

            // Show simple progress animation
            const progressContainer = document.getElementById('progress-container');
            const progressBar = document.getElementById('progress-bar');

            if (progressContainer) {
                progressContainer.style.display = 'block';
            }

            if (progressBar) {
                progressBar.style.width = '0%';

                // Animate progress
                let progress = 0;
                const interval = setInterval(() => {
                    progress += 5;
                    progressBar.style.width = `${Math.min(progress, 90)}%`;
                    if (progress >= 90) clearInterval(interval);
                }, 50);
            }

            // Process each file
            uploadedImages = [];
            let processed = 0;

            fileArray.forEach(file => {
                const reader = new FileReader();

                reader.onload = function(e) {
                    // Create image data object
                    const imageData = {
                        name: file.name,
                        type: file.type,
                        size: file.size,
                        dataUrl: e.target.result
                    };

                    // Add to array
                    uploadedImages.push(imageData);

                    // Update progress
                    processed++;
                    if (processed === fileArray.length) {
                        // Complete progress bar
                        if (progressBar) {
                            progressBar.style.width = '100%';
                        }

                        // Display image previews
                        showImagePreviews();

                        // Hide progress after a short delay
                        setTimeout(() => {
                            if (progressContainer) {
                                progressContainer.style.display = 'none';
                            }
                        }, 300);

                        showToast(`${uploadedImages.length} image(s) uploaded successfully`, 'success');
                    }
                };

                reader.onerror = function() {
                    console.error('Error reading file:', file.name);
                    processed++;

                    if (processed === fileArray.length) {
                        if (progressContainer) {
                            progressContainer.style.display = 'none';
                        }
                        showImagePreviews();
                    }
                };

                // Start reading file
                reader.readAsDataURL(file);
            });
        }

        // Show preview of uploaded images
        function showImagePreviews() {
            console.log('Showing image previews for', uploadedImages.length, 'images');

            const uploadPreview = document.getElementById('upload-preview');
            if (!uploadPreview) {
                console.error('Upload preview container not found');
                return;
            }

            // Clear preview area
            uploadPreview.innerHTML = '';

            // Add styles to upload preview
            uploadPreview.style.display = 'flex';
            uploadPreview.style.flexWrap = 'wrap';
            uploadPreview.style.gap = '10px';
            uploadPreview.style.marginTop = '10px';

            // Display message if no images
            if (!uploadedImages || uploadedImages.length === 0) {
                console.log('No images to preview');
                const noImagesMsg = document.createElement('div');
                noImagesMsg.className = 'no-images-message';
                noImagesMsg.textContent = 'No images uploaded yet';
                noImagesMsg.style.width = '100%';
                noImagesMsg.style.textAlign = 'center';
                noImagesMsg.style.padding = '15px';
                noImagesMsg.style.color = '#999';
                uploadPreview.appendChild(noImagesMsg);
                return;
            }

            // Add each image to the preview
            uploadedImages.forEach((image, index) => {
                const previewItem = document.createElement('div');
                previewItem.className = 'preview-item';
                previewItem.style.position = 'relative';
                previewItem.style.flex = '0 0 calc(33.333% - 10px)';
                previewItem.style.maxWidth = '150px';
                previewItem.style.marginBottom = '10px';

                // Create image element
                const img = document.createElement('img');
                img.src = image.dataUrl;
                img.alt = `Image ${index + 1}`;
                img.className = 'preview-image';
                img.style.width = '100%';
                img.style.height = 'auto';
                img.style.borderRadius = '4px';
                img.style.objectFit = 'cover';
                img.style.aspectRatio = '1/1';

                // Create remove button
                const removeBtn = document.createElement('button');
                removeBtn.className = 'remove-image-btn';
                removeBtn.innerHTML = '<i class="fas fa-times"></i>';
                removeBtn.style.position = 'absolute';
                removeBtn.style.top = '5px';
                removeBtn.style.right = '5px';
                removeBtn.style.background = 'rgba(0,0,0,0.5)';
                removeBtn.style.color = 'white';
                removeBtn.style.border = 'none';
                removeBtn.style.borderRadius = '50%';
                removeBtn.style.width = '20px';
                removeBtn.style.height = '20px';
                removeBtn.style.display = 'flex';
                removeBtn.style.alignItems = 'center';
                removeBtn.style.justifyContent = 'center';
                removeBtn.style.cursor = 'pointer';
                removeBtn.title = 'Remove image';

                removeBtn.onclick = function(e) {
                    e.preventDefault();
                    e.stopPropagation();

                    // Remove from array
                    uploadedImages.splice(index, 1);

                    // Update preview
                    showImagePreviews();

                    // Show feedback
                    showToast('Image removed', 'info');
                };

                // Add to preview item
                previewItem.appendChild(img);
                previewItem.appendChild(removeBtn);

                // Add to preview container
                uploadPreview.appendChild(previewItem);
            });

            // Update button states
            const generateBtn = document.getElementById('generate-btn');
            const browseBtn = document.getElementById('browse-btn');

            if (generateBtn) {
                generateBtn.disabled = uploadedImages.length === 0 && isImageMode;
            }

            if (browseBtn) {
                browseBtn.disabled = uploadedImages.length >= 5;
                if (uploadedImages.length >= 5) {
                    showToast('Maximum of 5 images reached', 'info');
                }
            }
        }

        // Handle story generation
        function handleGenerateStory(event) {
            // Prevent default form submission
            if (event) event.preventDefault();

            console.log('Generating story, mode:', isImageMode ? 'image' : 'text');

            // Show loading overlay
            const loadingOverlay = document.getElementById('loading-overlay');
            if (loadingOverlay) loadingOverlay.style.display = 'flex';

            // Get form values
            const title = document.getElementById('story-title')?.value.trim() || '';
            const subtitle = document.getElementById('story-subtitle')?.value.trim() || '';
            const textPrompt = document.getElementById('text-prompt')?.value.trim() || '';

            // Validate based on current mode
            if (!title) {
                showToast('Please enter a story title', 'error');
                if (loadingOverlay) loadingOverlay.style.display = 'none';
                return;
            }

            if (isImageMode) {
                // Image mode validation
                if (uploadedImages.length === 0) {
                    showToast('Please upload at least one image', 'error');
                    if (loadingOverlay) loadingOverlay.style.display = 'none';
                    return;
                }
            } else {
                // Text mode validation
                if (!textPrompt) {
                    showToast('Please enter a description for your story', 'error');
                    if (loadingOverlay) loadingOverlay.style.display = 'none';
                    return;
                }

                if (textPrompt.length < 10) {
                    showToast('Please enter a more detailed description for better results', 'error');
                    if (loadingOverlay) loadingOverlay.style.display = 'none';
                    return;
                }
            }

            // Get the selected style
            const styleOptions = document.querySelectorAll('input[name="style"]');
            let selectedStyle = 'fantasy';
            styleOptions.forEach(option => {
                if (option.checked) {
                    selectedStyle = option.value;
                }
            });

            // Generate story
            try {
                // In a real app, this would be an API call
                setTimeout(() => {
                    console.log('Story generated with:', {
                        mode: isImageMode ? 'image' : 'text',
                        style: selectedStyle,
                        images: isImageMode ? uploadedImages.length : 0,
                        textPrompt: !isImageMode ? textPrompt : 'N/A'
                    });

                    // Create a mock story object
                    const story = {
                        title: title,
                        subtitle: subtitle || `A ${selectedStyle} Story`,
                        style: selectedStyle,
                        intro: `Once upon a time in a ${selectedStyle} world...`,
                        content: [
                            `This is the first paragraph of our ${selectedStyle} story.`,
                            `This is the second paragraph about ${title}.`,
                            `This is the third paragraph with more ${selectedStyle} elements.`,
                            !isImageMode ? `Inspired by: "${textPrompt}"` : 'A magical adventure continues...',
                            `The story builds to its ${selectedStyle} climax.`
                        ],
                        conclusion: `And that concludes our tale of ${title}.`,
                        images: isImageMode ? uploadedImages : []
                    };

                    // Update the book with the story
                    updateBookWithStory(story);

                    // Show the result
                    const storyResult = document.getElementById('story-result');
                    if (storyResult) {
                        storyResult.style.display = 'flex';
                        storyResult.scrollIntoView({ behavior: 'smooth' });
                    }

                    // Hide loading overlay
                    if (loadingOverlay) loadingOverlay.style.display = 'none';

                    showToast('Story generated successfully!', 'success');
                }, 1500);
            } catch (error) {
                console.error('Error generating story:', error);
                showToast('Failed to generate story', 'error');
                if (loadingOverlay) loadingOverlay.style.display = 'none';
            }
        }

        // Process the story response
        function processStoryResponse(response) {
            console.log('Processing story response');
            const loadingOverlay = document.getElementById('loading-overlay');
            if (loadingOverlay) loadingOverlay.style.display = 'none';

            const result = document.getElementById('story-result');
            if (result) result.style.display = 'block';

            if (!response || !response.story) {
                showToast('Failed to generate story. Please try again.', 'error');
                return;
            }

            // Update book content
            updateBookWithStory(response.story);

            // Scroll to results
            setTimeout(() => {
                const resultsArea = document.getElementById('story-result');
                if (resultsArea) {
                    resultsArea.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
            }, 100);

            // Animate book opening
            animateBookOpening();
        }

        // Update book with story content
        function updateBookWithStory(story) {
            console.log('Updating book with story');

            // Update cover title and subtitle
            const coverTitle = document.getElementById('cover-title');
            const coverSubtitle = document.getElementById('cover-subtitle');
            const bookSpine = document.getElementById('book-spine');

            if (coverTitle) coverTitle.textContent = story.title;
            if (coverSubtitle) coverSubtitle.textContent = story.subtitle || 'A Tale of Imagination';
            if (bookSpine) bookSpine.textContent = story.title;

            // Set random book color
            document.documentElement.style.setProperty('--book-color', getRandomBookColor());
            document.documentElement.style.setProperty('--book-spine', getRandomBookColor(0.8));
            document.documentElement.style.setProperty('--page-color', '#f8f5e6');

            // Create pages
            const bookPages = document.getElementById('book-pages');
            if (!bookPages) {
                console.error('Book pages container not found');
                return;
            }

            // Clear existing content
            bookPages.innerHTML = '';

            // Calculate how many page spreads we need
            const pagesContent = [story.intro, ...story.content, story.conclusion];

            // Create page spreads (2 pages side by side)
            const totalSpreads = Math.ceil(pagesContent.length / 2);
            let contentIndex = 0;

            console.log(`Creating ${totalSpreads} page spreads for ${pagesContent.length} content sections`);

            for (let i = 0; i < totalSpreads; i++) {
                // Create a spread (two pages side by side)
                const spread = document.createElement('div');
                spread.className = 'book-page-spread';
                spread.dataset.page = i + 1;
                spread.style.display = i === 0 ? 'flex' : 'none'; // Show only first spread

                // Left page
                const leftPage = document.createElement('div');
                leftPage.className = 'page-left';

                if (contentIndex < pagesContent.length) {
                    const leftContent = document.createElement('div');
                    leftContent.className = 'page-content';
                    leftContent.innerHTML = `<p>${pagesContent[contentIndex]}</p>`;
                    leftPage.appendChild(leftContent);
                    contentIndex++;
                }

                // Right page
                const rightPage = document.createElement('div');
                rightPage.className = 'page-right';

                if (contentIndex < pagesContent.length) {
                    const rightContent = document.createElement('div');
                    rightContent.className = 'page-content';
                    rightContent.innerHTML = `<p>${pagesContent[contentIndex]}</p>`;
                    rightPage.appendChild(rightContent);
                    contentIndex++;
                }

                // Add pages to spread
                spread.appendChild(leftPage);
                spread.appendChild(rightPage);

                // Add spread to book
                bookPages.appendChild(spread);
            }

            // Update page navigation
            updatePageNavigation(totalSpreads);

            // Add image if available
            if (story.images && story.images.length > 0) {
                // Add the first image to the first right page
                const firstRightPage = bookPages.querySelector('.book-page-spread[data-page="1"] .page-right');
                if (firstRightPage && firstRightPage.firstChild) {
                    const imageElement = document.createElement('img');
                    imageElement.src = story.images[0].dataUrl;
                    imageElement.className = 'story-image';
                    imageElement.style.width = '100%';
                    imageElement.style.marginBottom = '15px';
                    imageElement.style.borderRadius = '4px';
                    imageElement.style.boxShadow = '0 2px 4px rgba(0,0,0,0.1)';

                    // Insert at beginning of the page
                    firstRightPage.insertBefore(imageElement, firstRightPage.firstChild);
                }
            }
        }

        // Animate the book opening
        function animateBookOpening() {
            const book = document.getElementById('book-3d');
            const cover = document.getElementById('book-cover');
            
            // First add the classes to initiate animations
            book.classList.add('book-open');
            
            // Wait a bit then open the cover
            setTimeout(() => {
                cover.classList.add('open');
                
                // After cover is open, update page content
                setTimeout(() => {
                    updatePageNavigation(currentStory.spreads.length);
                    goToPage(0); // Show first page
                }, 800);
            }, 400);
            
            // Add a subtle wobble animation when book opens
            book.animate([
                { transform: 'rotateY(0deg) translateY(0)' },
                { transform: 'rotateY(0deg) translateY(-10px)' },
                { transform: 'rotateY(0deg) translateY(0)' }
            ], {
                duration: 800,
                delay: 800,
                easing: 'cubic-bezier(0.34, 1.56, 0.64, 1)'
            });
        }

        // Function to get a random book color
        function getRandomBookColor(opacity = 1) {
            const bookColors = [
                `rgba(139, 69, 19, ${opacity})`,    // Saddle Brown
                `rgba(128, 0, 0, ${opacity})`,      // Maroon
                `rgba(25, 25, 112, ${opacity})`,    // Midnight Blue
                `rgba(0, 100, 0, ${opacity})`,      // Dark Green
                `rgba(72, 61, 139, ${opacity})`,    // Dark Slate Blue
                `rgba(47, 79, 79, ${opacity})`,     // Dark Slate Gray
                `rgba(85, 107, 47, ${opacity})`,    // Dark Olive Green
                `rgba(178, 34, 34, ${opacity})`,    // Firebrick
                `rgba(139, 0, 0, ${opacity})`,      // Dark Red
                `rgba(85, 45, 25, ${opacity})`      // Rich Brown
            ];
            return bookColors[Math.floor(Math.random() * bookColors.length)];
        }

        // Function to show toast
        function showToast(message, type = 'info') {
            const toastContainer = document.getElementById('toast-container');
            if (toastContainer) {
                const toast = document.createElement('div');
                toast.className = `toast ${type}`;
                toast.textContent = message;
                toastContainer.appendChild(toast);

                // Auto-remove after 3 seconds
                setTimeout(() => {
                    toast.classList.add('hide');
                    setTimeout(() => {
                        toastContainer.removeChild(toast);
                    }, 300);
                }, 3000);
            }
        }

        function animatePageTurn(currentPageElement, nextPageElement) {
            // Get elements
            const pageAnimation = document.getElementById('page-turn-animation');
            
            // Set up initial state
            if (currentPageElement) {
                // Clone current page content to animation layer
                pageAnimation.innerHTML = currentPageElement.innerHTML;
                pageAnimation.style.display = 'block';
                
                // Animate the page turn
                pageAnimation.animate([
                    { transform: 'rotateY(0deg)', transformOrigin: 'left center', boxShadow: '-5px 0 10px rgba(0,0,0,0.1)' },
                    { transform: 'rotateY(-90deg)', transformOrigin: 'left center', boxShadow: '-5px 0 15px rgba(0,0,0,0.2)' }
                ], {
                    duration: 800,
                    easing: 'cubic-bezier(0.645, 0.045, 0.355, 1)'
                }).onfinish = () => {
                    // Hide animation layer
                    pageAnimation.style.display = 'none';
                    pageAnimation.innerHTML = '';
                    
                    // Show next page with a nice fade-in
                    if (nextPageElement) {
                        nextPageElement.style.display = 'block';
                        nextPageElement.animate([
                            { opacity: 0, transform: 'translateX(20px)' },
                            { opacity: 1, transform: 'translateX(0)' }
                        ], {
                            duration: 400,
                            easing: 'ease-out'
                        });
                    }
                };
            } else if (nextPageElement) {
                // If there's no current page (first page), just show the next page
                nextPageElement.style.display = 'block';
                nextPageElement.animate([
                    { opacity: 0 },
                    { opacity: 1 }
                ], {
                    duration: 400,
                    easing: 'ease-out'
                });
            }
        }

        // Modify existing functions to use the new animations
        function openBook() {
            const book = document.getElementById('book-3d');
            const readBtn = document.getElementById('read-btn');
            
            if (book.classList.contains('book-closed')) {
                book.classList.remove('book-closed');
                animateBookOpening();
                readBtn.innerHTML = '<i class="fas fa-book"></i> Close Book';
            } else {
                closeBook();
                readBtn.innerHTML = '<i class="fas fa-book-open"></i> Open Book';
            }
        }

        function goToPage(index) {
            const pages = document.querySelectorAll('.book-page');
            const currentPage = document.getElementById('current-page');
            const prevBtn = document.getElementById('prev-page-btn');
            const nextBtn = document.getElementById('next-page-btn');
            
            // Hide all pages
            pages.forEach(page => page.style.display = 'none');
            
            // Get current and next page elements
            const currentPageElement = currentPageIndex >= 0 ? pages[currentPageIndex] : null;
            const nextPageElement = pages[index];
            
            // Animate the page turn
            animatePageTurn(currentPageElement, nextPageElement);
            
            // Update navigation
            currentPageIndex = index;
            currentPage.textContent = index + 1;
            
            // Update button states
            prevBtn.disabled = index === 0;
            nextBtn.disabled = index === pages.length - 1;
        }
    </script>
</body>
</html>
