<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1.0">
  <title>{{ title }}</title>

  <!-- Bootstrap & your CSS -->
  <link rel="stylesheet" href="/static/css/bootstrap.min.css">
  <link rel="stylesheet" href="/static/css/styles.css">

  <style>
    /* popup overlay & box */
    #overlay {
      display: none;
      position: fixed; top: 0; left: 0;
      width:100vw; height:100vh;
      background: rgba(0,0,0,0.5);
      z-index: 9998;
    }
    #popup {
      display: none;
      position: fixed; top:50%; left:50%;
      transform: translate(-50%,-50%);
      background: white;
      border: 2px solid #333;
      border-radius: 8px;
      padding: 20px;
      max-width: 90vw; max-height: 80vh;
      overflow: hidden;
      z-index: 9999;
      display: flex; flex-direction: column;
    }
    #popup-body {
      flex: 1;            /* take all leftover space */
      overflow-y: auto;   /* vertical scroll */
      text-align: left;
      padding-right: 8px; /* room for scrollbar */
    }
    #popup-footer {
      margin-top: 12px;
      text-align: center;
    }
  </style>
</head>
<body>

  <!-- NAVBAR -->
  <header>
    <nav id="nav-content" class="navbar navbar-expand-sm"></nav>
    <script>$("#nav-content").load("/static/html/navbar_admin.html");</script>
  </header>

  <main class="container my-4">

    <h3><u>{{ title }}</u></h3>
    <ul style="font-size: 1rem;">
      <li>Please choose a Workbook CSV file to upload</li>
      <li>Only one file may be uploaded per day</li>
      <li>File should not be empty</li>
      <li>The assessment file once uploaded will be sent to MFT</li>
    </ul>

    <div class="upload-box my-3">
      <form id="uploadForm" method="POST" action="/assessment" enctype="multipart/form-data">
        <input type="file" id="fileInput" name="file_upload" accept=".csv">
        <button type="submit" class="btn btn-primary ml-2">Upload</button>
      </form>
    </div>

    <!-- ONLY if message exists and is non-empty -->
    {% if message is defined and message not in [None, '', 'None'] %}
      <!-- hidden container for the HTML table or error message -->
      <div id="hidden-table" style="display:none;">
        {{ message | safe }}
      </div>
      <script>
        document.addEventListener('DOMContentLoaded', () => {
          const html = document.getElementById('hidden-table').innerHTML.trim();
          if (html) {
            showPopup('', html);
          }
        });
      </script>
    {% endif %}

    <!-- Popup overlay & box -->
    <div id="overlay"></div>
    <div id="popup">
      <div id="popup-body"></div>
      <div id="popup-footer">
        <button class="btn btn-secondary" onclick="closePopup()">Close</button>
      </div>
    </div>

    <!-- FILES TABLE -->
    {% if files %}
      <h4 class="mt-5">Files</h4>
      <form id="fileListForm" method="POST" action="/download-multiple">
        <table class="table table-bordered">
          <thead>
            <tr>
              <th>Select</th><th>File Name</th><th>Creation Time</th><th>Created By</th><th>Download</th>
            </tr>
          </thead>
          <tbody>
            {% for f in files %}
              <tr>
                <td><input type="checkbox" name="file_names" value="{{ f.name }}"></td>
                <td>{{ f.name }}</td>
                <td>{{ f.creation_time }}</td>
                <td>{{ f.created_by }}</td>
                <td><a href="/download/{{ f.name }}">Download</a></td>
              </tr>
            {% endfor %}
          </tbody>
        </table>
        <button class="btn btn-primary">Download Selected</button>
      </form>
    {% endif %}

  </main>

  <footer class="text-center py-3">
    &copy; Business Engineering | <a href="mailto:icen_be_analytics@reddinse.com">Contact BE</a> | <a href="#">Back to top</a>
  </footer>

  <!-- JS -->
  <script src="/static/js/jquery.min.js"></script>
  <script src="/static/js/bootstrap.min.js"></script>
  <script>
    function showPopup(msg, html) {
      document.getElementById('popup-body').innerHTML = html || msg;
      document.getElementById('overlay').style.display = 'block';
      document.getElementById('popup').style.display   = 'flex';
    }
    function closePopup() {
      document.getElementById('overlay').style.display = 'none';
      document.getElementById('popup').style.display   = 'none';
    }

    // Prevent submit with no file
    document.getElementById('uploadForm')
      .addEventListener('submit', e => {
        if (!document.getElementById('fileInput').files.length) {
          e.preventDefault();
          showPopup('Error','Please select the file to upload.');
        }
      });
  </script>
</body>
</html>
