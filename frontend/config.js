/* Where the page finds its API. The frontend container rewrites this file at start-up from the
   API_BASE environment variable, so the page and the backend are joined only by this setting
   (a path on the same host, or a full address). Default for development: the same host. */
window.CHURNSENSE_CONFIG = { apiBase: "/api/v1" };
