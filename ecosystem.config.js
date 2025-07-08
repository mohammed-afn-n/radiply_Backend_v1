module.exports = {
  apps: [
  
    {
      name: "celery_worker",
      script: "celery",
      args: "-A radiplybackend worker --loglevel=info",
      interpreter: "python3"
    },
    {
      name: "celery_beat",
      script: "celery",
      args: "-A radiplybackend beat --loglevel=info",
      interpreter: "python3"
    }
  ]
}