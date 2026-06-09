(function () {
  function applyTheme(mode) {
    var selected = mode || localStorage.getItem('douchi-theme') || 'auto';
    var dark = selected === 'dark' || (selected === 'auto' && window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches);
    document.documentElement.classList.toggle('dark', dark);
    var iframe = document.querySelector('iframe.giscus-frame');
    if (iframe) {
      iframe.contentWindow.postMessage({ giscus: { setConfig: { theme: dark ? 'dark' : 'light' } } }, 'https://giscus.app');
    }
  }

  document.addEventListener('DOMContentLoaded', function () {
    document.querySelectorAll('[data-theme-toggle]').forEach(function (button) {
      button.addEventListener('click', function () {
        var isDark = document.documentElement.classList.contains('dark');
        var next = isDark ? 'light' : 'dark';
        localStorage.setItem('douchi-theme', next);
        applyTheme(next);
      });
    });

    var toggle = document.querySelector('.sidebar-toggle');
    var panel = document.querySelector('.left-panel');
    if (toggle && panel) {
      toggle.addEventListener('click', function () {
        var open = panel.classList.toggle('is-open');
        toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
      });
    }

    document.querySelectorAll('.code-copy').forEach(function (button) {
      button.addEventListener('click', function () {
        var block = button.closest('.code-block');
        var code = block && block.querySelector('code');
        if (!code || !navigator.clipboard) return;
        navigator.clipboard.writeText(code.innerText).then(function () {
          var old = button.textContent;
          button.textContent = '已复制';
          setTimeout(function () { button.textContent = old; }, 1200);
        });
      });
    });

    var input = document.getElementById('site-search');
    var results = document.getElementById('search-results');
    var indexPromise;
    if (input && results) {
      input.addEventListener('input', function () {
        var q = input.value.trim().toLowerCase();
        if (!q) {
          results.hidden = true;
          results.innerHTML = '';
          return;
        }
        if (!indexPromise) {
          indexPromise = fetch('/index.json').then(function (res) { return res.json(); }).catch(function () { return []; });
        }
        indexPromise.then(function (items) {
          var matches = items.filter(function (item) {
            var haystack = [item.title, item.summary, item.content, (item.tags || []).join(' '), (item.categories || []).join(' ')].join(' ').toLowerCase();
            return haystack.indexOf(q) !== -1;
          }).slice(0, 8);
          results.innerHTML = matches.length ? matches.map(function (item) {
            return '<a class="search-result" href="' + item.permalink + '"><strong>' + escapeHtml(item.title) + '</strong><small>' + escapeHtml((item.summary || '').slice(0, 70)) + '</small></a>';
          }).join('') : '<div class="search-result"><small>没有找到结果</small></div>';
          results.hidden = false;
        });
      });
      document.addEventListener('click', function (event) {
        if (!results.contains(event.target) && event.target !== input) results.hidden = true;
      });
    }

    initTocAutoExpand();
    initArticleImageLightbox();
  });

  function initTocAutoExpand() {
    var toc = document.getElementById('TableOfContents');
    var tocBox = document.getElementById('toc-content-auto');
    var article = document.querySelector('.article-content');
    if (!toc || !tocBox || !article) return;

    var links = Array.prototype.slice.call(toc.querySelectorAll('a[href^="#"]'));
    var headings = links.map(function (link) {
      var id = decodeURIComponent(link.getAttribute('href').slice(1));
      var heading = document.getElementById(id);
      return heading ? { link: link, heading: heading } : null;
    }).filter(Boolean);
    var items = Array.prototype.slice.call(toc.querySelectorAll('li'));
    if (!headings.length) return;

    function clearState() {
      links.forEach(function (link) { link.classList.remove('active'); });
      items.forEach(function (item) {
        item.classList.remove('has-active');
        item.classList.remove('is-active');
      });
    }

    function markActive(link) {
      clearState();
      link.classList.add('active');
      var node = link.parentElement;
      if (node) node.classList.add('is-active');
      while (node && node !== toc) {
        if (node.tagName && node.tagName.toLowerCase() === 'li') node.classList.add('has-active');
        node = node.parentElement;
      }
      var activeTop = link.offsetTop;
      if (activeTop < tocBox.scrollTop || activeTop > tocBox.scrollTop + tocBox.clientHeight - 48) {
        tocBox.scrollTop = Math.max(0, activeTop - 80);
      }
    }

    function updateToc() {
      var offset = 96;
      var active = headings[0];
      for (var i = 0; i < headings.length; i += 1) {
        if (headings[i].heading.getBoundingClientRect().top <= offset) active = headings[i];
        else break;
      }
      markActive(active.link);
    }

    updateToc();
    window.addEventListener('scroll', updateToc, { passive: true });
    window.addEventListener('resize', updateToc);
  }

  function initArticleImageLightbox() {
    var lightbox = document.querySelector('[data-article-lightbox]');
    if (!lightbox) return;
    var image = lightbox.querySelector('[data-article-lightbox-image]');
    var caption = lightbox.querySelector('[data-article-lightbox-caption]');
    var close = lightbox.querySelector('[data-article-lightbox-close]');

    function openLightbox(src, alt, title) {
      image.src = src;
      image.alt = alt || title || '';
      caption.textContent = title || alt || '';
      caption.hidden = !(title || alt);
      lightbox.classList.add('is-open');
      lightbox.setAttribute('aria-hidden', 'false');
      document.documentElement.style.overflow = 'hidden';
    }

    function closeLightbox() {
      lightbox.classList.remove('is-open');
      lightbox.setAttribute('aria-hidden', 'true');
      image.src = '';
      document.documentElement.style.overflow = '';
    }

    document.addEventListener('click', function (event) {
      var button = event.target.closest && event.target.closest('[data-article-image-src]');
      if (button) {
        openLightbox(
          button.getAttribute('data-article-image-src'),
          button.getAttribute('data-article-image-alt'),
          button.getAttribute('data-article-image-title')
        );
        return;
      }
      if (event.target === lightbox || event.target === close) closeLightbox();
    });

    document.addEventListener('keydown', function (event) {
      if (event.key === 'Escape' && lightbox.classList.contains('is-open')) closeLightbox();
    });
  }

  function escapeHtml(str) {
    return String(str || '').replace(/[&<>\'\"]/g, function (ch) {
      return ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' })[ch];
    });
  }
})();
