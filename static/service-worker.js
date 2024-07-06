// static/service-worker.js

const CACHE_NAME = 'my-flask-pwa-v1';
const CACHE_URLS = [
    '/',
    '/offline.html',
    '/static/css/detect.css',
    '/static/css/index.css',
    '/static/css/predict.css',
    '/static/css/recipes.css',
    '/static/js/detect.js',
    '/templates/Images',  // This assumes you want to cache the Images folder as a whole
    '/templates/detect.html',
    '/templates/index.html',
    '/templates/predict.html',
    '/templates/recipeHome.html',
    '/templates/recipes.html'
];

// Install Service Worker
self.addEventListener('install', event => {
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then(cache => {
                console.log('Cache opened');
                return cache.addAll(CACHE_URLS);
            })
            .catch(error => {
                console.error('Error caching files:', error);
            })
    );
});

// Activate Service Worker
self.addEventListener('activate', event => {
    event.waitUntil(
        caches.keys().then(cacheNames => {
            return Promise.all(
                cacheNames.map(cache => {
                    if (cache !== CACHE_NAME) {
                        console.log('Deleting old cache:', cache);
                        return caches.delete(cache);
                    }
                })
            );
        })
    );
});

// Fetch from Service Worker
self.addEventListener('fetch', event => {
    event.respondWith(
        caches.match(event.request)
            .then(response => {
                return response || fetch(event.request);
            })
            .catch(() => {
                if (event.request.mode === 'navigate') {
                    return caches.match('/offline.html');
                }
            })
    );
});
